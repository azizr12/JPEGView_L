#include "stdafx.h"

#include "JXLWrapper.h"
#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/resizable_parallel_runner.h"
#include "jxl/resizable_parallel_runner_cxx.h"
#include "MaxImageDef.h"
#include "ICCProfileTransform.h"

static bool IsTiffHeader(const uint8_t* p) {
	return (p[0] == 'I' && p[1] == 'I' && p[2] == 0x2A && p[3] == 0x00) ||
		(p[0] == 'M' && p[1] == 'M' && p[2] == 0x00 && p[3] == 0x2A);
}

// The Exif box payload starts with a 4-byte big endian offset to the TIFF header.
// Some encoders also (incorrectly) include the "Exif\0\0" prefix, so handle both.
static void GetJxlTiffData(const std::vector<uint8_t>& exif, const uint8_t*& tiff, size_t& tiff_size) {
	tiff = NULL;
	tiff_size = 0;
	if (exif.size() < 8)
		return;

	const uint8_t* p = exif.data();
	size_t n = exif.size();
	if (n >= 14 && !memcmp(p, "Exif\0\0", 6)) {
		tiff = p + 6;
		tiff_size = n - 6;
		return;
	}

	size_t candidates[3];
	int count = 0;
	size_t offset = ((size_t)p[0] << 24) | ((size_t)p[1] << 16) | ((size_t)p[2] << 8) | p[3];
	// Offsets are relative to the byte after the offset field, but also accept
	// offsets relative to the payload start.
	if (4 + offset + 8 <= n)
		candidates[count++] = 4 + offset;
	if (offset + 8 <= n && offset != 4 + offset)
		candidates[count++] = offset;
	candidates[count++] = 0;

	for (int i = 0; i < count; i++) {
		size_t start = candidates[i];
		if (IsTiffHeader(p + start)) {
			tiff = p + start;
			tiff_size = n - start;
			return;
		}
	}
}

// Reads the Exif box of a JPEG XL container without decoding the image. This has
// to be a separate pass because the Exif box can be located before the codestream
// (and therefore before any image data is available to the main decoder).
static bool ReadJxlExif(const uint8_t* jxl_data, size_t size, std::vector<uint8_t>& exif) {
	JxlDecoderPtr decoder = JxlDecoderMake(nullptr);
	if (!decoder)
		return false;
	if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(decoder.get(), JXL_DEC_BOX | JXL_DEC_BOX_COMPLETE))
		return false;
	JxlDecoderSetDecompressBoxes(decoder.get(), JXL_TRUE);
	JxlDecoderSetInput(decoder.get(), jxl_data, size);
	JxlDecoderCloseInput(decoder.get());

	const size_t kChunkSize = 65536;
	size_t output_pos = 0;
	bool reading_exif = false;

	for (;;) {
		JxlDecoderStatus status = JxlDecoderProcessInput(decoder.get());
		if (status == JXL_DEC_ERROR || status == JXL_DEC_NEED_MORE_INPUT) {
			return false;
		} else if (status == JXL_DEC_BOX) {
			if (reading_exif) {
				size_t remaining = JxlDecoderReleaseBoxBuffer(decoder.get());
				exif.resize(exif.size() - remaining);
				return !exif.empty();
			}
			JxlBoxType type;
			if (JXL_DEC_SUCCESS != JxlDecoderGetBoxType(decoder.get(), type, JXL_TRUE))
				return false;
			if (!memcmp(type, "Exif", 4)) {
				reading_exif = true;
				exif.resize(kChunkSize);
				JxlDecoderSetBoxBuffer(decoder.get(), exif.data(), exif.size());
			}
		} else if (status == JXL_DEC_BOX_NEED_MORE_OUTPUT) {
			if (!reading_exif)
				return false;
			size_t remaining = JxlDecoderReleaseBoxBuffer(decoder.get());
			output_pos += kChunkSize - remaining;
			exif.resize(exif.size() + kChunkSize);
			JxlDecoderSetBoxBuffer(decoder.get(), exif.data() + output_pos, exif.size() - output_pos);
		} else if (status == JXL_DEC_BOX_COMPLETE) {
			if (reading_exif) {
				size_t remaining = JxlDecoderReleaseBoxBuffer(decoder.get());
				exif.resize(exif.size() - remaining);
				return !exif.empty();
			}
		} else if (status == JXL_DEC_SUCCESS) {
			return false;
		} else {
			return false;
		}
	}
}

static int CountJxlFrames(const uint8_t* jxl_data, size_t size) {
    auto decoder = JxlDecoderMake(nullptr);
    if (!decoder) return 1;

    // Wir abonnieren NUR das Frame-Event (keine Pixel, keine Farbprofile, keine Boxen)
    if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(decoder.get(), JXL_DEC_FRAME)) {
        return 1;
    }

    JxlDecoderSetInput(decoder.get(), jxl_data, size);
    JxlDecoderCloseInput(decoder.get());

    int count = 0;
    for (;;) {
        JxlDecoderStatus status = JxlDecoderProcessInput(decoder.get());
        if (status == JXL_DEC_FRAME) {
            count++;
        } else if (status == JXL_DEC_SUCCESS || status == JXL_DEC_ERROR || status == JXL_DEC_NEED_MORE_INPUT) {
            break;
        }
    }
    return count > 0 ? count : 1;
}

struct JxlReader::jxl_cache {
	JxlDecoderPtr decoder;
	JxlResizableParallelRunnerPtr runner;
	JxlBasicInfo info;
	uint8_t* data;
	size_t data_size;
	int prev_frame_timestamp;
	int width;
	int height;
	void* transform;
	std::vector<uint8_t> exif;
	bool exif_read;
	int total_frames;
};

JxlReader::jxl_cache JxlReader::cache = { 0 };

// based on https://github.com/libjxl/libjxl/blob/main/examples/decode_oneshot.cc
// and https://github.com/libjxl/libjxl/blob/main/examples/decode_exif_metadata.cc
bool JxlReader::DecodeJpegXlOneShot(const uint8_t* jxl, size_t size, std::vector<uint8_t>* pixels, int& xsize,
	int& ysize, bool& have_animation, int& frame_count, int& frame_time, std::vector<uint8_t>* icc_profile, bool& outOfMemory) {

	if (cache.decoder.get() == NULL) {
		cache.runner = JxlResizableParallelRunnerMake(nullptr);

		cache.decoder = JxlDecoderMake(nullptr);
		if (JXL_DEC_SUCCESS !=
			JxlDecoderSubscribeEvents(cache.decoder.get(), JXL_DEC_BASIC_INFO |
				JXL_DEC_COLOR_ENCODING |
				JXL_DEC_FRAME |
				JXL_DEC_FULL_IMAGE)) {
			return false;
		}

		if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(cache.decoder.get(),
			JxlResizableParallelRunner,
			cache.runner.get())) {
			return false;
		}

		JxlDecoderSetInput(cache.decoder.get(), jxl, size);
		JxlDecoderCloseInput(cache.decoder.get());
		cache.data = (uint8_t*)jxl;
		cache.data_size = size;
	}

	JxlBasicInfo info;
	JxlPixelFormat format = { 4, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0 };

	bool loop_check = false;
	for (;;) {
		JxlDecoderStatus status = JxlDecoderProcessInput(cache.decoder.get());

		if (status == JXL_DEC_ERROR) {
			return false;
		} else if (status == JXL_DEC_NEED_MORE_INPUT) {
			return false;
		} else if (status == JXL_DEC_BASIC_INFO) {
			if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(cache.decoder.get(), &info)) {
				return false;
			}
			cache.info = info;
			if (cache.info.xsize > MAX_IMAGE_DIMENSION || cache.info.ysize > MAX_IMAGE_DIMENSION)
				return false;
			if (abs((double)cache.info.xsize * cache.info.ysize) > MAX_IMAGE_PIXELS) {
				outOfMemory = true;
				return false;
			}
			JxlResizableParallelRunnerSetThreads(
				cache.runner.get(),
				JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));
		} else if (status == JXL_DEC_COLOR_ENCODING) {
			// Get the ICC color profile of the pixel data
			size_t icc_size;
			if (JXL_DEC_SUCCESS !=
				JxlDecoderGetICCProfileSize(
					cache.decoder.get(), JXL_COLOR_PROFILE_TARGET_DATA, &icc_size)) {
				return false;
			}
			icc_profile->resize(icc_size);
			if (JXL_DEC_SUCCESS != JxlDecoderGetColorAsICCProfile(
				cache.decoder.get(),
				JXL_COLOR_PROFILE_TARGET_DATA,
				icc_profile->data(), icc_profile->size())) {
				return false;
			}
		} else if (status == JXL_DEC_FRAME) {
			JxlFrameHeader header;
			JxlDecoderGetFrameHeader(cache.decoder.get(), &header);
			JxlAnimationHeader animation = cache.info.animation;
			if (animation.tps_numerator)
				frame_time = (int)(1000.0 * header.duration * animation.tps_denominator / animation.tps_numerator);
			else
				frame_time = 0;
		} else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
			size_t buffer_size;
			if (JXL_DEC_SUCCESS !=
				JxlDecoderImageOutBufferSize(cache.decoder.get(), &format, &buffer_size)) {
				return false;
			}
			if (buffer_size != cache.info.xsize * cache.info.ysize * 4) {
				return false;
			}
			pixels->resize(cache.info.xsize * cache.info.ysize * 4);
			void* pixels_buffer = static_cast<void*>(pixels->data());
			size_t pixels_buffer_size = pixels->size() * sizeof(uint8_t);
			if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(cache.decoder.get(), &format,
				pixels_buffer,
				pixels_buffer_size)) {
				return false;
			}
		} else if (status == JXL_DEC_FULL_IMAGE) {
			// Full frame has been decoded

			xsize = cache.info.xsize;
			ysize = cache.info.ysize;
			have_animation = cache.info.have_animation;
			if (have_animation) {
				// Falls die Frame-Anzahl für diese Datei noch nicht ermittelt wurde: einmalig zählen
				if (cache.total_frames == 0) {
					cache.total_frames = CountJxlFrames(cache.data, cache.data_size);
				}
				frame_count = cache.total_frames;
			} else {
				frame_count = 1;
			}
			return true;
		} else if (status == JXL_DEC_SUCCESS) {
			// All decoding successfully finished, we are at the end of the file.
			// We must rewind the decoder to get a new frame.
			if (loop_check)
				return false; // escape infinite loop
			loop_check = true;
			JxlDecoderRewind(cache.decoder.get());
			JxlDecoderSubscribeEvents(cache.decoder.get(), JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE);
			JxlDecoderSetInput(cache.decoder.get(), cache.data, cache.data_size);
			JxlDecoderCloseInput(cache.decoder.get());
		} else {
			return false;
		}
	}
}

void* JxlReader::ReadImage(int& width,
	int& height,
	int& nchannels,
	bool& has_animation,
	int& frame_count,
	int& frame_time,
	void*& exif_chunk,
	bool& outOfMemory,
	const void* buffer,
	int sizebytes)
{
	outOfMemory = false;
	width = height = 0;
	nchannels = 4;
	has_animation = false;
	unsigned char* pPixelData = NULL;
	exif_chunk = NULL;


	std::vector<uint8_t> pixels;
	std::vector<uint8_t> icc_profile;
	if (!DecodeJpegXlOneShot((const uint8_t*)buffer, sizebytes, &pixels, width, height,
		has_animation, frame_count, frame_time, &icc_profile, outOfMemory)) {
		return NULL;
	}
	size_t size = (size_t)width * height * nchannels;
	pPixelData = new(std::nothrow) unsigned char[size];
	if (pPixelData == NULL) {
		outOfMemory = true;
		return NULL;
	}
	if (cache.transform == NULL)
		cache.transform = ICCProfileTransform::CreateTransform(icc_profile.data(), icc_profile.size(), ICCProfileTransform::FORMAT_RGBA);
	if (!ICCProfileTransform::DoTransform(cache.transform, pixels.data(), pPixelData, width, height)) {
		// RGBA -> BGRA conversion (with little-endian integers)
		uint32_t* data = (uint32_t*)pixels.data();
		for (size_t i = 0; i * sizeof(uint32_t) < size; i++) {
			((uint32_t*)pPixelData)[i] = _rotr(_byteswap_ulong(data[i]), 8);
		}
	}

	// Copy Exif data into the format understood by CEXIFReader
	if (!cache.exif_read) {
		ReadJxlExif((const uint8_t*)buffer, sizebytes, cache.exif);
		cache.exif_read = true;
	}
	const uint8_t* tiff = NULL;
	size_t tiff_size = 0;
	GetJxlTiffData(cache.exif, tiff, tiff_size);
	if (tiff != NULL && tiff_size > 8 && tiff_size + 8 <= 65533) {
		exif_chunk = malloc(tiff_size + 10);
		if (exif_chunk != NULL) {
			uint8_t* chunk = (uint8_t*)exif_chunk;
			uint16_t app1_size = _byteswap_ushort((uint16_t)(tiff_size + 8));
			chunk[0] = 0xFF;
			chunk[1] = 0xE1;
			memcpy(chunk + 2, &app1_size, 2);
			memcpy(chunk + 4, "Exif\0\0", 6);
			memcpy(chunk + 10, tiff, tiff_size);
		}
	}

	if (!has_animation)
		DeleteCache();

	return pPixelData;
}

void JxlReader::DeleteCache() {
	// Do NOT free cache.data at this point! 
	// It is a buffer allocated in CImageLoadThread.
	// CImageLoadThread is responsible for freeing it.
	
	ICCProfileTransform::DeleteTransform(cache.transform);
	// Setting the decoder and runner to 0 (NULL) will automatically destroy them
	cache = { 0 };
}

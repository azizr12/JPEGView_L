#include "stdafx.h"

#include "JXLWrapper.h"
#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/resizable_parallel_runner.h"
#include "jxl/resizable_parallel_runner_cxx.h"
#include "MaxImageDef.h"
#include "ICCProfileTransform.h"

// Decoder and thread pool live for the whole app and are only reset between images.
// (Same single-thread assumption as JxlReader::cache.)
static JxlDecoderPtr g_decoder;
static JxlResizableParallelRunnerPtr g_runner;

// Optional fast-first-paint preview (see JxlReader::SetPreviewCallback)
static JxlReader::PreviewCallback g_preview_cb = NULL;
static void* g_preview_user = NULL;
static size_t g_preview_min_pixels = 0;

void JxlReader::SetPreviewCallback(PreviewCallback cb, void* user, size_t min_pixels) {
	g_preview_cb = cb;
	g_preview_user = user;
	g_preview_min_pixels = min_pixels;
}

// RGBA -> BGRA conversion (with little-endian integers), src == dst is allowed
static void RgbaToBgra(const uint8_t* src, uint8_t* dst, size_t npix) {
	const uint32_t* s = (const uint32_t*)src;
	uint32_t* d = (uint32_t*)dst;
	for (size_t i = 0; i < npix; i++)
		d[i] = _rotr(_byteswap_ulong(s[i]), 8);
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

// True if the data color profile is plain sRGB (no color transform needed)
static bool IsPlainSrgb(JxlDecoder* dec) {
	JxlColorEncoding enc;
	// Fails if the data profile can only be expressed as ICC
	if (JXL_DEC_SUCCESS != JxlDecoderGetColorAsEncodedProfile(dec, JXL_COLOR_PROFILE_TARGET_DATA, &enc))
		return false;
	return enc.color_space == JXL_COLOR_SPACE_RGB
		&& enc.white_point == JXL_WHITE_POINT_D65
		&& enc.primaries == JXL_PRIMARIES_SRGB
		&& enc.transfer_function == JXL_TRANSFER_FUNCTION_SRGB;
}

// Reads only the header (basic info + color encoding) with a throw-away decoder,
// so the real decoder can subscribe to exactly the events this file needs.
static bool ProbeJxl(const uint8_t* jxl, size_t size, JxlBasicInfo& info, bool& plain_srgb,
	std::vector<uint8_t>* icc_profile) {
	auto dec = JxlDecoderMake(nullptr);
	if (!dec) return false;
	if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING))
		return false;
	JxlDecoderSetInput(dec.get(), jxl, size);
	JxlDecoderCloseInput(dec.get());

	bool got_info = false, got_color = false;
	while (!(got_info && got_color)) {
		JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
		if (status == JXL_DEC_BASIC_INFO) {
			if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec.get(), &info))
				return false;
			got_info = true;
		} else if (status == JXL_DEC_COLOR_ENCODING) {
			plain_srgb = IsPlainSrgb(dec.get());
			if (!plain_srgb) {
				// Get the ICC color profile of the pixel data (only needed when not plain sRGB)
				size_t icc_size;
				if (JXL_DEC_SUCCESS !=
					JxlDecoderGetICCProfileSize(dec.get(), JXL_COLOR_PROFILE_TARGET_DATA, &icc_size)) {
					return false;
				}
				icc_profile->resize(icc_size);
				if (JXL_DEC_SUCCESS != JxlDecoderGetColorAsICCProfile(
					dec.get(),
					JXL_COLOR_PROFILE_TARGET_DATA,
					icc_profile->data(), icc_profile->size())) {
					return false;
				}
			}
			got_color = true;
		} else {
			return false;
		}
	}
	return true;
}

struct JxlReader::jxl_cache {
	bool active; // true while a file is set up in g_decoder (kept across animation frames)
	JxlBasicInfo info;
	uint8_t* data;
	size_t data_size;
	int prev_frame_timestamp;
	int width;
	int height;
	void* transform;
	std::vector<uint8_t> exif;
	int total_frames;
	bool plain_srgb;
};

JxlReader::jxl_cache JxlReader::cache = { 0 };

// based on https://github.com/libjxl/libjxl/blob/main/examples/decode_oneshot.cc
// and https://github.com/libjxl/libjxl/blob/main/examples/decode_exif_metadata.cc
bool JxlReader::DecodeJpegXlOneShot(const uint8_t* jxl, size_t size, uint8_t** pixels, int& xsize,
	int& ysize, bool& have_animation, int& frame_count, int& frame_time, std::vector<uint8_t>* icc_profile, bool& outOfMemory) {

	if (!cache.active) {
		// 1. Transfer ownership of the buffer IMMEDIATELY.
		// This guarantees that if any subsequent setup step fails and returns false, 
		// DeleteCache() will still correctly free the buffer.
		cache.data = (uint8_t*)jxl;
		cache.data_size = size;

		// Probe the header first, then subscribe only to what this file needs
		if (!ProbeJxl(jxl, size, cache.info, cache.plain_srgb, icc_profile))
			return false;
            
		if (cache.info.xsize > MAX_IMAGE_DIMENSION || cache.info.ysize > MAX_IMAGE_DIMENSION)
			return false;
            
		if ((double)cache.info.xsize * cache.info.ysize > MAX_IMAGE_PIXELS) {
			outOfMemory = true;
			return false;
		}

		// Thread pool is created once and only resized afterwards
		if (g_runner.get() == NULL) {
			g_runner = JxlResizableParallelRunnerMake(nullptr);
			if (g_runner.get() == NULL)
				return false;
		}
		JxlResizableParallelRunnerSetThreads(
			g_runner.get(),
			JxlResizableParallelRunnerSuggestThreads(cache.info.xsize, cache.info.ysize));

		// Decoder is created once and reset for every new file
		if (g_decoder.get() == NULL) {
			g_decoder = JxlDecoderMake(nullptr);
			if (g_decoder.get() == NULL)
				return false;
		} else {
			JxlDecoderReset(g_decoder.get());
		}

		int events = JXL_DEC_FULL_IMAGE;
		if (cache.info.have_animation)
			events |= JXL_DEC_FRAME;
            
		// A bare codestream cannot contain Exif, only the container can
		bool container = JxlSignatureCheck(jxl, size) == JXL_SIG_CONTAINER;
		if (container)
			events |= JXL_DEC_BOX;
            
		// Fast first paint: for big stills get a callback as soon as the DC (1:8) data is decoded
		bool want_preview = g_preview_cb != NULL && !cache.info.have_animation &&
			(size_t)cache.info.xsize * cache.info.ysize >= g_preview_min_pixels;
		if (want_preview)
			events |= JXL_DEC_FRAME_PROGRESSION;

		if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(g_decoder.get(), events)) {
			return false;
		}

		if (want_preview) {
			if (JXL_DEC_SUCCESS != JxlDecoderSetProgressiveDetail(g_decoder.get(), kDC)) {
				return false;
			}
		}

		if (container) {
			if (JXL_DEC_SUCCESS != JxlDecoderSetDecompressBoxes(g_decoder.get(), JXL_TRUE)) {
				return false;
			}
		}

		if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(g_decoder.get(),
			JxlResizableParallelRunner,
			g_runner.get())) {
			return false;
		}

		// The ICC transform is needed by the preview too, so create it up front
		if (!cache.plain_srgb && cache.transform == NULL)
			cache.transform = ICCProfileTransform::CreateTransform(icc_profile->data(), icc_profile->size(), ICCProfileTransform::FORMAT_RGBA);

		JxlDecoderSetInput(g_decoder.get(), jxl, size);
		JxlDecoderCloseInput(g_decoder.get());
		
		// 2. Mark cache as active (cache.data and cache.data_size were already set at the top)
		cache.active = true;
	}

	JxlDecoder* dec = g_decoder.get();
	JxlPixelFormat format = { 4, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0 };
	const constexpr size_t kChunkSize = 65536;
	size_t output_pos = 0;

	bool preview_sent = false;
	bool loop_check = false;
	for (;;) {
		JxlDecoderStatus status = JxlDecoderProcessInput(dec);

		if (status == JXL_DEC_ERROR) {
			return false;
		} else if (status == JXL_DEC_NEED_MORE_INPUT) {
			return false;
		} else if (status == JXL_DEC_FRAME) {
			JxlFrameHeader header;
			JxlDecoderGetFrameHeader(dec, &header);
			JxlAnimationHeader animation = cache.info.animation;
			if (animation.tps_numerator)
				frame_time = (int)(1000.0 * header.duration * animation.tps_denominator / animation.tps_numerator);
			else
				frame_time = 0;
		} else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
			size_t buffer_size;
			if (JXL_DEC_SUCCESS !=
				JxlDecoderImageOutBufferSize(dec, &format, &buffer_size)) {
				return false;
			}
			if (buffer_size != (size_t)cache.info.xsize * cache.info.ysize * 4) {
				return false;
			}
			// Decode straight into the final buffer (new[] does not zero-fill)
			delete[] *pixels;
			*pixels = new(std::nothrow) uint8_t[buffer_size];
			if (*pixels == NULL) {
				outOfMemory = true;
				return false;
			}
			if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(dec, &format,
				*pixels,
				buffer_size)) {
				return false;
			}
		} else if (status == JXL_DEC_FRAME_PROGRESSION) {
			// DC data is decoded: flush a blurry full-size image into the out buffer and hand it
			// to the viewer while the real decode continues. The final decode overwrites
			// the whole buffer, so converting it in place here is fine.
			if (!preview_sent && *pixels != NULL && g_preview_cb != NULL) {
				preview_sent = true;
				if (JXL_DEC_SUCCESS == JxlDecoderFlushImage(dec)) {
					int w = (int)cache.info.xsize, h = (int)cache.info.ysize;
					size_t npix = (size_t)w * h;
					if (cache.plain_srgb) {
						RgbaToBgra(*pixels, *pixels, npix);
						g_preview_cb(*pixels, w, h, g_preview_user);
					} else {
						uint8_t* tmp = new(std::nothrow) uint8_t[npix * 4];
						if (tmp != NULL) {
							if (!ICCProfileTransform::DoTransform(cache.transform, *pixels, tmp, w, h))
								RgbaToBgra(*pixels, tmp, npix);
							g_preview_cb(tmp, w, h, g_preview_user);
							delete[] tmp;
						}
					}
				}
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
			JxlDecoderRewind(dec);
			JxlDecoderSubscribeEvents(dec, JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE);
			JxlDecoderSetInput(dec, cache.data, cache.data_size);
			JxlDecoderCloseInput(dec);
		} else if (status == JXL_DEC_BOX) {
			if (!cache.exif.empty()) {
				size_t remaining = JxlDecoderReleaseBoxBuffer(dec);
				cache.exif.resize(cache.exif.size() - remaining);
			} else {
				JxlBoxType type;
				if (JXL_DEC_SUCCESS !=
					JxlDecoderGetBoxType(dec, type, true)) {
					return false;
				}
				if (!memcmp(type, "Exif", 4)) {
					cache.exif.resize(kChunkSize);
					JxlDecoderSetBoxBuffer(dec, cache.exif.data(), cache.exif.size());
				}
			}
		} else if (status == JXL_DEC_BOX_NEED_MORE_OUTPUT) {
			size_t remaining = JxlDecoderReleaseBoxBuffer(dec);
			output_pos += kChunkSize - remaining;
			cache.exif.resize(cache.exif.size() + kChunkSize);
			JxlDecoderSetBoxBuffer(dec, cache.exif.data() + output_pos, cache.exif.size() - output_pos);
		} else if (status == JXL_DEC_BOX_COMPLETE) {
			if (!cache.exif.empty()) {
				size_t remaining = JxlDecoderReleaseBoxBuffer(dec);
				cache.exif.resize(cache.exif.size() - remaining);
				
				// DO NOT return true here. 
				// The metadata box is finished, but the decoder must continue 
				// processing the stream to reach the image payload (JXL_DEC_FULL_IMAGE).
			}
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
	frame_time = 0;
	unsigned char* pPixelData = NULL;
	exif_chunk = NULL;


	uint8_t* pixels = NULL; // decoded RGBA, allocated by DecodeJpegXlOneShot
	std::vector<uint8_t> icc_profile;
	if (!DecodeJpegXlOneShot((const uint8_t*)buffer, sizebytes, &pixels, width, height,
		has_animation, frame_count, frame_time, &icc_profile, outOfMemory)) {
		delete[] pixels;
		DeleteCache(); // don't leave a half-used decoder for the next file
		return NULL;
	}
	if (pixels == NULL) {
		DeleteCache();
		return NULL;
	}
	size_t size = (size_t)width * height * nchannels;
	size_t npix = (size_t)width * height;

	if (cache.plain_srgb) {
		// in-place RGBA -> BGRA
		RgbaToBgra(pixels, pixels, npix);
		pPixelData = pixels;
	} else {
		pPixelData = new(std::nothrow) unsigned char[size];
		if (pPixelData == NULL) {
			delete[] pixels;
			outOfMemory = true;
			DeleteCache();
			return NULL;
		}
		if (!ICCProfileTransform::DoTransform(cache.transform, pixels, pPixelData, width, height)) {
			RgbaToBgra(pixels, pPixelData, npix);
		}
		delete[] pixels;
	}

	// Copy Exif data into the format understood by CEXIFReader
	if (!cache.exif.empty() && cache.exif.size() > 8 && cache.exif.size() < 65532) {
		exif_chunk = malloc(cache.exif.size() + 6);
		if (exif_chunk != NULL) {
			memcpy(exif_chunk, "\xFF\xE1\0\0Exif\0\0", 10);
			*((unsigned short*)exif_chunk + 1) = _byteswap_ushort(cache.exif.size() + 4);
			memcpy((uint8_t*)exif_chunk + 10, cache.exif.data() + 4, cache.exif.size() - 4);
		}
	}

	if (!has_animation)
		DeleteCache();

	return pPixelData;
}

void JxlReader::DeleteCache() {
	// The wrapper now takes ownership of the buffer and frees it here.
	if (cache.data != NULL) {
		delete[] cache.data;
		cache.data = NULL;
	}

	if (cache.transform != NULL) {
		ICCProfileTransform::DeleteTransform(cache.transform);
		cache.transform = NULL;
	}

	// The decoder and the thread pool are kept alive for the next image.
	// Reset releases the memory held by the last image.
	if (g_decoder.get() != NULL)
		JxlDecoderReset(g_decoder.get());
		
	cache = {}; // Properly resets the struct and clears the std::vector
}
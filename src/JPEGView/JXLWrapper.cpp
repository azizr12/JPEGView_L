#include "stdafx.h"

#include "JXLWrapper.h"
#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/resizable_parallel_runner.h"
#include "jxl/resizable_parallel_runner_cxx.h"
#include "MaxImageDef.h"
#include "ICCProfileTransform.h"
#include "SettingsProvider.h"
#include <memory>
#include <thread>

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define JXL_HAVE_SSE2 1
#else
#define JXL_HAVE_SSE2 0
#endif

static bool IsSrgbColorEncoding(const JxlColorEncoding& encoding) {
	return encoding.color_space == JXL_COLOR_SPACE_RGB &&
		encoding.white_point == JXL_WHITE_POINT_D65 &&
		encoding.primaries == JXL_PRIMARIES_SRGB &&
		encoding.transfer_function == JXL_TRANSFER_FUNCTION_SRGB;
}

// JPEGView expects BGRA, libjxl produces RGBA. Swap R and B, four pixels at a time.
static void ConvertRgbaToBgra(const uint8_t* source, uint8_t* target, size_t numPixels) {
#if JXL_HAVE_SSE2
	const __m128i maskRB = _mm_set1_epi32(static_cast<int>(0x00FF00FFu));
	const __m128i maskGA = _mm_set1_epi32(static_cast<int>(0xFF00FF00u));
	size_t i = 0;
	for (; i + 4 <= numPixels; i += 4) {
		__m128i value = _mm_loadu_si128((const __m128i*)(source + i * 4));
		__m128i rb = _mm_and_si128(value, maskRB);
		rb = _mm_or_si128(_mm_slli_epi32(rb, 16), _mm_srli_epi32(rb, 16));
		__m128i ga = _mm_and_si128(value, maskGA);
		_mm_storeu_si128((__m128i*)(target + i * 4), _mm_or_si128(ga, rb));
	}
	for (; i < numPixels; ++i) {
		const uint8_t* s = source + i * 4;
		uint8_t* t = target + i * 4;
		t[0] = s[2]; t[1] = s[1]; t[2] = s[0]; t[3] = s[3];
	}
#else
	for (size_t i = 0; i < numPixels; ++i) {
		const uint8_t* s = source + i * 4;
		uint8_t* t = target + i * 4;
		t[0] = s[2]; t[1] = s[1]; t[2] = s[0]; t[3] = s[3];
	}
#endif
}

static bool SwapRedBlueRange(uint8_t* data, size_t begin, size_t end) {
#if JXL_HAVE_SSE2
	const __m128i maskRB = _mm_set1_epi32(static_cast<int>(0x00FF00FFu));
	const __m128i maskGA = _mm_set1_epi32(static_cast<int>(0xFF00FF00u));
	const __m128i alphaMask = _mm_set1_epi32(static_cast<int>(0xFF000000u));
	__m128i nonOpaque = _mm_setzero_si128();
	size_t i = begin;
	for (; i + 4 <= end; i += 4) {
		__m128i value = _mm_loadu_si128((const __m128i*)(data + i * 4));
		nonOpaque = _mm_or_si128(nonOpaque, _mm_xor_si128(_mm_and_si128(value, alphaMask), alphaMask));
		__m128i rb = _mm_and_si128(value, maskRB);
		rb = _mm_or_si128(_mm_slli_epi32(rb, 16), _mm_srli_epi32(rb, 16));
		__m128i ga = _mm_and_si128(value, maskGA);
		_mm_storeu_si128((__m128i*)(data + i * 4), _mm_or_si128(ga, rb));
	}
	bool allOpaque = _mm_movemask_epi8(_mm_cmpeq_epi8(nonOpaque, _mm_setzero_si128())) == 0xFFFF;
	for (; i < end; ++i) {
		uint8_t* p = data + i * 4;
		if (p[3] != 0xFF) allOpaque = false;
		uint8_t temp = p[0]; p[0] = p[2]; p[2] = temp;
	}
	return allOpaque;
#else
	bool allOpaque = true;
	for (size_t i = begin; i < end; ++i) {
		uint8_t* p = data + i * 4;
		if (p[3] != 0xFF) allOpaque = false;
		uint8_t temp = p[0]; p[0] = p[2]; p[2] = temp;
	}
	return allOpaque;
#endif
}

static bool SwapRedBlueInPlace(uint8_t* data, size_t numPixels) {
	const size_t parallelThreshold = 8u * 1024 * 1024;
	unsigned threads = 1;
	if (numPixels >= parallelThreshold) {
		const unsigned cores = std::thread::hardware_concurrency();
		threads = cores > 4 ? 4 : (cores == 0 ? 1 : cores);
	}
	if (threads <= 1) {
		return SwapRedBlueRange(data, 0, numPixels);
	}
	std::vector<unsigned char> opaque(threads, 0);
	std::vector<std::thread> workers;
	workers.reserve(threads);
	const size_t chunk = (numPixels + threads - 1) / threads;
	for (unsigned t = 0; t < threads; ++t) {
		const size_t begin = chunk * t;
		if (begin >= numPixels) {
			break;
		}
		size_t end = begin + chunk;
		if (end > numPixels) {
			end = numPixels;
		}
		workers.emplace_back([data, begin, end, &opaque, t]() {
			opaque[t] = SwapRedBlueRange(data, begin, end) ? 1 : 0;
		});
	}
	for (size_t i = 0; i < workers.size(); ++i) {
		workers[i].join();
	}
	for (size_t i = 0; i < workers.size(); ++i) {
		if (opaque[i] == 0) {
			return false;
		}
	}
	return true;
}

static bool IsAllOpaqueRange(const uint8_t* data, size_t begin, size_t end) {
#if JXL_HAVE_SSE2
	const __m128i alphaMask = _mm_set1_epi32(static_cast<int>(0xFF000000u));
	__m128i nonOpaque = _mm_setzero_si128();
	size_t i = begin;
	for (; i + 4 <= end; i += 4) {
		__m128i value = _mm_loadu_si128((const __m128i*)(data + i * 4));
		nonOpaque = _mm_or_si128(nonOpaque, _mm_xor_si128(_mm_and_si128(value, alphaMask), alphaMask));
	}
	if (_mm_movemask_epi8(_mm_cmpeq_epi8(nonOpaque, _mm_setzero_si128())) != 0xFFFF) {
		return false;
	}
	for (; i < end; ++i) {
		if (data[i * 4 + 3] != 0xFF) return false;
	}
	return true;
#else
	for (size_t i = begin; i < end; ++i) {
		if (data[i * 4 + 3] != 0xFF) return false;
	}
	return true;
#endif
}

static bool IsAllOpaque(const uint8_t* data, size_t numPixels) {
	const size_t parallelThreshold = 8u * 1024 * 1024;
	unsigned threads = 1;
	if (numPixels >= parallelThreshold) {
		const unsigned cores = std::thread::hardware_concurrency();
		threads = cores > 4 ? 4 : (cores == 0 ? 1 : cores);
	}
	if (threads <= 1) {
		return IsAllOpaqueRange(data, 0, numPixels);
	}
	std::vector<unsigned char> opaque(threads, 0);
	std::vector<std::thread> workers;
	workers.reserve(threads);
	const size_t chunk = (numPixels + threads - 1) / threads;
	for (unsigned t = 0; t < threads; ++t) {
		const size_t begin = chunk * t;
		if (begin >= numPixels) {
			break;
		}
		size_t end = begin + chunk;
		if (end > numPixels) {
			end = numPixels;
		}
		workers.emplace_back([data, begin, end, &opaque, t]() {
			opaque[t] = IsAllOpaqueRange(data, begin, end) ? 1 : 0;
		});
	}
	for (size_t i = 0; i < workers.size(); ++i) {
		workers[i].join();
	}
	for (size_t i = 0; i < workers.size(); ++i) {
		if (opaque[i] == 0) {
			return false;
		}
	}
	return true;
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
	std::vector<void*> extraTransforms;
	std::vector<uint8_t> exif;
	int total_frames;
	bool color_profile_processed;
};

JxlReader::jxl_cache JxlReader::cache = {};

// based on https://github.com/libjxl/libjxl/blob/main/examples/decode_oneshot.cc
// and https://github.com/libjxl/libjxl/blob/main/examples/decode_exif_metadata.cc
bool JxlReader::DecodeJpegXlOneShot(const uint8_t* jxl, size_t size, std::unique_ptr<uint8_t[]>* pixels, int& xsize,
	int& ysize, bool& have_animation, int& frame_count, int& frame_time, std::vector<uint8_t>* icc_profile,
	bool& outOfMemory, unsigned char** outPixels) {

	*outPixels = NULL;
	std::unique_ptr<unsigned char[]> directBuffer;
	bool useDirectBuffer = cache.color_profile_processed && cache.transform == NULL;

	if (cache.decoder.get() == NULL) {
		// Keep the parallel runner alive across images: it owns the worker threads.
		if (cache.runner.get() == NULL) {
			cache.runner = JxlResizableParallelRunnerMake(nullptr);
		}

		cache.decoder = JxlDecoderMake(nullptr);
		if (JXL_DEC_SUCCESS !=
			JxlDecoderSubscribeEvents(cache.decoder.get(), JXL_DEC_BASIC_INFO |
				JXL_DEC_COLOR_ENCODING |
				JXL_DEC_BOX |
				JXL_DEC_FRAME |
				JXL_DEC_FULL_IMAGE)) {
			return false;
		}

		if (JXL_DEC_SUCCESS != JxlDecoderSetDecompressBoxes(cache.decoder.get(), JXL_TRUE)) {
			return false;
		}

		if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(cache.decoder.get(),
			JxlResizableParallelRunner,
			cache.runner.get())) {
			return false;
		}

		cache.color_profile_processed = false;

		JxlDecoderSetInput(cache.decoder.get(), jxl, size);
		JxlDecoderCloseInput(cache.decoder.get());
		cache.data = (uint8_t*)jxl;
		cache.data_size = size;
	}

	JxlBasicInfo info;
	JxlPixelFormat format = { 4, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0 };
	const constexpr size_t kChunkSize = 65536;
	size_t output_pos = 0;

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
			bool isSrgb = false;
			if (cache.transform == NULL) {
				JxlColorEncoding color_encoding;
				isSrgb = JXL_DEC_SUCCESS == JxlDecoderGetColorAsEncodedProfile(
					cache.decoder.get(), JXL_COLOR_PROFILE_TARGET_DATA, &color_encoding) &&
					IsSrgbColorEncoding(color_encoding);
			}

			if (!isSrgb) {
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

				if (cache.transform == NULL) {
					cache.transform = ICCProfileTransform::CreateTransform(
						icc_profile->data(), icc_profile->size(), ICCProfileTransform::FORMAT_RGBA);
					if (cache.transform != NULL &&
						CSettingsProvider::This().UseEmbeddedColorProfiles() &&
						(size_t)cache.info.xsize * cache.info.ysize >= 2u * 1024 * 1024 &&
						icc_profile->size() <= 65536) {
						unsigned cores = std::thread::hardware_concurrency();
						if (cores > 4) cores = 4;
						for (unsigned i = 1; i < cores; ++i) {
							void* extra = ICCProfileTransform::CreateTransform(
								icc_profile->data(), icc_profile->size(), ICCProfileTransform::FORMAT_RGBA);
							if (extra == NULL) break;
							cache.extraTransforms.push_back(extra);
						}
					}
				}
			}

			cache.color_profile_processed = true;
			// No transform means the decoder output is already sRGB and can be written
			// straight into the final buffer, skipping the lcms pass.
			useDirectBuffer = cache.transform == NULL;
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
			if (buffer_size != (size_t)cache.info.xsize * cache.info.ysize * 4) {
				return false;
			}
			if (useDirectBuffer) {
				// No color conversion needed: decode straight into the final buffer.
				directBuffer.reset(new(std::nothrow) unsigned char[buffer_size]);
				if (directBuffer.get() == NULL) {
					outOfMemory = true;
					return false;
				}
				if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(cache.decoder.get(), &format,
					directBuffer.get(), buffer_size)) {
					return false;
				}
			} else {
				pixels->reset(new(std::nothrow) uint8_t[buffer_size]);
				if (pixels->get() == NULL) {
					outOfMemory = true;
					return false;
				}
				if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(cache.decoder.get(), &format,
					static_cast<void*>(pixels->get()), buffer_size)) {
					return false;
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
			if (useDirectBuffer) {
				*outPixels = directBuffer.release();
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
		} else if (status == JXL_DEC_BOX) {
			if (!cache.exif.empty()) {
				size_t remaining = JxlDecoderReleaseBoxBuffer(cache.decoder.get());
				cache.exif.resize(cache.exif.size() - remaining);
			} else {
				JxlBoxType type;
				if (JXL_DEC_SUCCESS !=
					JxlDecoderGetBoxType(cache.decoder.get(), type, true)) {
					return false;
				}
				if (!memcmp(type, "Exif", 4)) {
					cache.exif.resize(kChunkSize);
					JxlDecoderSetBoxBuffer(cache.decoder.get(), cache.exif.data(), cache.exif.size());
				}
			}
		} else if (status == JXL_DEC_BOX_NEED_MORE_OUTPUT) {
			size_t remaining = JxlDecoderReleaseBoxBuffer(cache.decoder.get());
			output_pos += kChunkSize - remaining;
			cache.exif.resize(cache.exif.size() + kChunkSize);
			JxlDecoderSetBoxBuffer(cache.decoder.get(), cache.exif.data() + output_pos, cache.exif.size() - output_pos);
		} else if (status == JXL_DEC_BOX_COMPLETE) {
			if (!cache.exif.empty()) {
				size_t remaining = JxlDecoderReleaseBoxBuffer(cache.decoder.get());
				cache.exif.resize(cache.exif.size() - remaining);
				return true;
			}
		} else {
			return false;
		}
	}
}

static bool TransformJxlPixels(void* transform, const std::vector<void*>& extraTransforms,
	const uint8_t* input, uint8_t* output, unsigned int width, unsigned int height) {
	if (transform == NULL) {
		return false;
	}
	const size_t threads = extraTransforms.size() + 1;
	if (threads == 1 || width == 0 || height == 0) {
		return ICCProfileTransform::DoTransform(transform, input, output, width, height);
	}
	std::vector<unsigned char> ok(threads, 0);
	std::vector<std::thread> workers;
	workers.reserve(threads);
	const unsigned int rowsPerThread = (height + (unsigned int)threads - 1) / (unsigned int)threads;
	for (size_t t = 0; t < threads; ++t) {
		const unsigned int begin = (unsigned int)(rowsPerThread * t);
		if (begin >= height) {
			break;
		}
		unsigned int end = begin + rowsPerThread;
		if (end > height) {
			end = height;
		}
		void* threadTransform = t == 0 ? transform : extraTransforms[t - 1];
		const size_t offset = (size_t)begin * width * 4;
		workers.emplace_back([=, &ok]() {
			ok[t] = ICCProfileTransform::DoTransform(threadTransform, input + offset,
				output + offset, width, end - begin) ? 1 : 0;
		});
	}
	for (size_t i = 0; i < workers.size(); ++i) {
		workers[i].join();
	}
	for (size_t i = 0; i < workers.size(); ++i) {
		if (ok[i] == 0) {
			return false;
		}
	}
	return true;
}

void* JxlReader::ReadImage(int& width,
	int& height,
	int& nchannels,
	bool& has_animation,
	int& frame_count,
	int& frame_time,
	void*& exif_chunk,
	bool& outOfMemory,
	bool& has_alpha,
	const void* buffer,
	int sizebytes)
{
	outOfMemory = false;
	width = height = 0;
	nchannels = 4;
	has_animation = false;
	has_alpha = false;
	unsigned char* pPixelData = NULL;
	exif_chunk = NULL;


	std::unique_ptr<uint8_t[]> pixels;
	std::vector<uint8_t> icc_profile;
	unsigned char* decodedPixels = NULL;
	if (!DecodeJpegXlOneShot((const uint8_t*)buffer, sizebytes, &pixels, width, height,
		has_animation, frame_count, frame_time, &icc_profile, outOfMemory, &decodedPixels)) {
		return NULL;
	}
	has_alpha = cache.info.alpha_bits > 0;
	size_t numPixels = (size_t)width * height;
	if (decodedPixels != NULL) {
		// The decoder wrote RGBA directly into the final buffer.
		if (SwapRedBlueInPlace(decodedPixels, numPixels) && has_alpha) {
			// Fully opaque alpha channel: Helpers::BlendAlpha would be a no-op.
			has_alpha = false;
		}
		pPixelData = decodedPixels;
	} else {
		pPixelData = new(std::nothrow) unsigned char[numPixels * nchannels];
		if (pPixelData == NULL) {
			outOfMemory = true;
			return NULL;
		}
		if (!TransformJxlPixels(cache.transform, cache.extraTransforms, pixels.get(), pPixelData, width, height)) {
			ConvertRgbaToBgra(pixels.get(), pPixelData, numPixels);
		}
		if (has_alpha && IsAllOpaque(pPixelData, numPixels)) {
			has_alpha = false;
		}
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
	// Do NOT free cache.data at this point! 
	// It is a buffer allocated in CImageLoadThread.
	// CImageLoadThread is responsible for freeing it.
	
	ICCProfileTransform::DeleteTransform(cache.transform);
	cache.transform = NULL;
	for (size_t i = 0; i < cache.extraTransforms.size(); ++i) {
		ICCProfileTransform::DeleteTransform(cache.extraTransforms[i]);
	}
	cache.extraTransforms.clear();
	cache.decoder.reset();
	cache.exif.clear();
	cache.total_frames = 0;
	cache.color_profile_processed = false;
	cache.data = NULL;
	cache.data_size = 0;
	cache.prev_frame_timestamp = 0;
	cache.width = 0;
	cache.height = 0;
	memset(&cache.info, 0, sizeof(cache.info));
	// Keep cache.runner alive: it owns the libjxl worker threads, which are
	// otherwise re-created for every single image.
}

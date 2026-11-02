#pragma once

#include <vector>

class JxlReader
{
public:
	// Called from inside ReadImage while a big still is still decoding.
	// bgra points to a blurry full-size preview (4 byte BGRA, width*height pixels).
	// The pointer is only valid during the call: copy the data, don't keep it.
	typedef void (*PreviewCallback)(const void* bgra, int width, int height, void* user);

	// Optional. Pass cb = NULL to disable. Only stills with at least min_pixels get a preview.
	static void SetPreviewCallback(PreviewCallback cb, void* user, size_t min_pixels = 4000000);

	// Returns data in 4 byte BGRA
	static void* ReadImage(int& width,   // width of the image loaded.
		int& height,  // height of the image loaded.
		int& bpp,     // BYTES (not bits) PER PIXEL.
		bool& has_animation,     // if the image is animated
		int& frame_count, // number of frames
		int& frame_time, // frame duration in milliseconds
		void*& exif, // Pointer to Exif data (must be freed by caller)
		bool& outOfMemory, // set to true when no memory to read image
		const void* buffer, // memory address containing jxl compressed data.
		int sizebytes); // size of jxl compressed data

	static void DeleteCache();

private:
	struct jxl_cache;
	static jxl_cache cache;
	static bool DecodeJpegXlOneShot(const uint8_t* jxl, size_t size, uint8_t** pixels, int& xsize,
		int& ysize, bool& have_animation, int& frame_count, int& frame_time, std::vector<uint8_t>* icc_profile, bool& outOfMemory);
};
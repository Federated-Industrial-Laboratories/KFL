/* pngout.h - write an RGB8 image as a PNG file, no libraries.
 *
 * The capture feature saves what the window shows, and a picture
 * format nothing can open is not a capture, so the format is PNG.
 * The encoder uses stored (uncompressed) deflate blocks: the files
 * are larger than a compressing encoder's, and in exchange the
 * whole writer is a page of arithmetic this tree owns, with no
 * dependency to vendor and nothing to misbehave. The output is
 * deterministic: the same pixels produce the same bytes.
 */
#ifndef K26RL_VIEW_PNGOUT_H
#define K26RL_VIEW_PNGOUT_H

#include <stdint.h>

namespace k26rl_view {

/* Write width x height RGB8 pixels (row-major, top row first, three
 * bytes per pixel) to path. Returns true on success; on failure the
 * partial file, if any, is removed. */
bool png_write_rgb8(const char *path, const uint8_t *pixels,
                    uint32_t width, uint32_t height);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_PNGOUT_H */

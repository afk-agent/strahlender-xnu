// Interface between the NVN executor (renderer.m) and the Metal plumbing (metal.m).
#pragma once
#ifdef __OBJC__
#import <Metal/Metal.h>
void renderer_init(id<MTLDevice> dev);
id<MTLCommandQueue> renderer_queue(void);
id<MTLTexture> renderer_texture(NvnTexture *t, id<MTLBlitCommandEncoder> blit, bool for_render);
MTLPixelFormat renderer_pixel_format(uint32_t nvn_format);
bool renderer_format_alpha_one(uint32_t nvn_format);
extern id<MTLBuffer> g_guest_buffer;
extern uint64_t g_guest_buffer_base;
// MetalFX temporal upscaler (taa.inc): what the present needs to scale the frame's presented image
typedef struct {
    bool active;                 // the frame was jittered
    __unsafe_unretained id<MTLTexture> scene;                 // the colour target the motion was written with (orientation check)
    __unsafe_unretained id<MTLTexture> motion, depth, copy;   // scene motion and depth (nil: none this frame); the presented
                                                              // image before the HUD (nil: nothing drawn over the scene)
    float jitter_x, jitter_y;    // content shift in pixels, y down in the scene's target
    bool depth_reversed;
} TaaFrame;
void renderer_taa_frame(id<MTLTexture> tex, uint32_t gw, uint32_t gh, TaaFrame *out);
void renderer_taa_status(bool ok, float out_scale);
#endif
#include <SDL3/SDL_events.h>

bool renderer_is_gpu_owned(NvnTexture *t);
void renderer_flush(void);  // finish all recorded GPU work
bool renderer_origin_lower_left(void);
const char *renderer_stats(void);
bool renderer_format_is_depth(uint32_t f);
bool renderer_format_has_stencil(uint32_t f);
void renderer_mark_gpu_owned(NvnTexture *t);
void input_handle_event(const SDL_Event *e);
bool input_update_text(SDL_Window *window, bool enabled);
bool input_user_blocked(const SDL_Event *e);

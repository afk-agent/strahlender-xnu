// Frame generation (setting "framegen"): the game runs at half the display's frame rate and Apple's MetalFX frame interpolator
// (macOS / iOS 26) makes the picture between two of its frames, which metal.m presents half a game frame before the newer one. The
// game's per-frame work - the processor's share, which keeps a passively cooled iPad warm - halves; the interpolation costs GPU time
// instead (runtime/test/metalfx_interp_test.m: 6 ms for a 1280x720 picture on an M1, 12 ms at 1920x1080). It needs the scene's motion
// and depth, which the renderer records for it (taa.inc), prepared by tfx_prep_encode (metalfx_temporal.h) with the motion set to zero
// under the HUD, so menus and text stay in place in the generated picture.
// Include once per program, after metalfx_temporal.h.
#pragma once
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

static id g_fi;                      // id<MTLFXFrameInterpolator> (macOS / iOS 26)
static id<MTLTexture> g_fi_out, g_fi_prev;
static NSUInteger g_fi_iw, g_fi_ih, g_fi_cw, g_fi_ch;
static MTLPixelFormat g_fi_fmt;
static uint64_t g_fi_frame;          // frame of g_fi_prev (0: none)
static bool g_fi_fresh;              // the interpolator has no history yet
static uint64_t g_fi_generated;      // (stats) pictures made

static bool metalfx_interp_supported(id<MTLDevice> dev) {
    static int supported = -1;
    if (supported < 0) {
        supported = 0;
        if (@available(macOS 26.0, iOS 26.0, *)) supported = [MTLFXFrameInterpolatorDescriptor supportsDevice:dev] ? 1 : 0;
        METALFX_LOG("[metalfx] frame interpolation %s\n", supported ? "available" : "not available (needs macOS / iOS 26 and a supported GPU)");
    }
    return supported == 1;
}

// The picture between the previous call's `color` and this one's, or nil when there is none: the first frame, a frame after a gap or
// with reset set (a scene cut, no motion), a size change. color: this frame's image (any size, a raw format view); depth, motion: this
// frame's (R32Float, RG16Float, previous - current in uv units) at their own size; jitter: the content's shift of the depth and motion in
// pixels; dt: seconds between the two frames.
static id<MTLTexture> metalfx_interp_encode(id<MTLDevice> dev, id<MTLCommandBuffer> cb, id<MTLTexture> color, id<MTLTexture> depth,
                                            id<MTLTexture> motion, float jx, float jy, bool depth_reversed, float dt, uint64_t frame,
                                            bool reset) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
        if (!color || !depth || !motion || !metalfx_interp_supported(dev)) return nil;
        // (set up again only for other sizes: a refused configuration is not retried every frame; a release clears the sizes)
        if (g_fi_iw != motion.width || g_fi_ih != motion.height || g_fi_cw != color.width || g_fi_ch != color.height ||
            g_fi_fmt != color.pixelFormat) {
            g_fi = nil; g_fi_out = nil; g_fi_prev = nil;   // (the old one goes before the new one is made)
            g_fi_iw = motion.width; g_fi_ih = motion.height; g_fi_cw = color.width; g_fi_ch = color.height; g_fi_fmt = color.pixelFormat;
            if (!metalfx_memory_ok("frame interpolation", (uint64_t)motion.width * motion.height * 60 + (uint64_t)color.width * color.height * 8)) return nil;
            size_t mem0 = dev.currentAllocatedSize;
            MTLFXFrameInterpolatorDescriptor *d = [MTLFXFrameInterpolatorDescriptor new];
            d.colorTextureFormat = color.pixelFormat;
            d.outputTextureFormat = color.pixelFormat;
            d.depthTextureFormat = MTLPixelFormatR32Float;
            d.motionTextureFormat = MTLPixelFormatRG16Float;
            d.uiTextureFormat = color.pixelFormat;
            d.inputWidth = motion.width; d.inputHeight = motion.height;
            d.outputWidth = color.width; d.outputHeight = color.height;
            id<MTLFXFrameInterpolator> fi = [d newFrameInterpolatorWithDevice:dev];
            g_fi = fi;
            g_fi_out = nil;
            if (fi) {
                MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:color.pixelFormat width:color.width
                                                                                            height:color.height mipmapped:NO];
                td.usage = fi.outputTextureUsage | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
                td.storageMode = MTLStorageModePrivate;
                g_fi_out = [dev newTextureWithDescriptor:td];
            }
            g_fi_fresh = true;
            METALFX_LOG("[metalfx] frame interpolation %lux%lu (motion %lux%lu): %s, %.0f MB\n", (unsigned long)color.width, (unsigned long)color.height,
                        (unsigned long)motion.width, (unsigned long)motion.height, fi && g_fi_out ? "ok" : "FAILED",
                        (double)((int64_t)dev.currentAllocatedSize - (int64_t)mem0) / 1048576.0);
            if (!g_fi_out) g_fi = nil;
        }
        if (!g_fi) return nil;
        id<MTLTexture> prev = g_fi_prev;
        bool usable = prev && !reset && frame == g_fi_frame + 1;
        g_fi_prev = color;
        g_fi_frame = frame;
        if (!usable) { g_fi_fresh = true; return nil; }
        id<MTLFXFrameInterpolator> fi = g_fi;
        fi.colorTexture = color;
        fi.prevColorTexture = prev;
        fi.depthTexture = depth;
        fi.motionTexture = motion;
        fi.outputTexture = g_fi_out;
        fi.motionVectorScaleX = (float)motion.width;
        fi.motionVectorScaleY = (float)motion.height;
        fi.jitterOffsetX = jx;
        fi.jitterOffsetY = jy;
        fi.deltaTime = dt;
        // camera values: the test above finds no difference between a 30 and a 60 degree field of view or other planes; the game's
        // overworld camera is about this
        fi.fieldOfView = 30;
        fi.aspectRatio = (float)color.width / (float)color.height;
        fi.nearPlane = 0.3f;
        fi.farPlane = 1000;
        fi.depthReversed = depth_reversed;
        fi.shouldResetHistory = g_fi_fresh;
        g_fi_fresh = false;
        [fi encodeToCommandBuffer:cb];
        g_fi_generated++;
        return g_fi_out;
    }
    return nil;
}

// Frame generation went off: the interpolator and its picture are released (about 57 bytes per input pixel).
static void metalfx_interp_release(void) {
    if (g_fi || g_fi_out) METALFX_LOG("[metalfx] frame interpolation released\n");
    g_fi = nil; g_fi_out = nil; g_fi_prev = nil;
    g_fi_iw = g_fi_ih = g_fi_cw = g_fi_ch = 0;
    g_fi_frame = 0;
}

// MetalFX spatial upscaling of the game's image before the present pass (metal.m present_gpu_locked). On when the setting
// "upscaler" is metalfx (BDSP_UPSCALER, read by nvn_glue.c settings_load; iPad Settings app, Mac View menu or config.ini) and
// the image is smaller than its place on the screen: a 720p or 1080p image on a 1440p window or an iPad's 1668-2064 pixel
// tall panel. Otherwise the present pass samples the image bilinearly, as before.
// Header only, so runtime/test/metalfx_scale_test.m runs this same code on a real image. Include it once per program
// (file-scope state); define METALFX_LOG before including it to route its two log lines.
#pragma once
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef METALFX_LOG
#include <stdio.h>
#define METALFX_LOG(...) fprintf(stderr, __VA_ARGS__)
#endif

static id<MTLFXSpatialScaler> g_fx;
static id<MTLTexture> g_fx_out;
static NSUInteger g_fx_iw, g_fx_ih, g_fx_ow, g_fx_oh;
static MTLPixelFormat g_fx_fmt;
static id<MTLTexture> g_fx_last;   // the texture (the view's parent) whose frame g_fx_out holds; nil: none
static uint64_t g_fx_last_frame;

// The image `in` is shown at ow_d x oh_d pixels: sets up the scaler for these sizes and returns its output texture, or nil when
// this is no upscale (plain sampling is right) or the GPU has no MetalFX.
static id<MTLTexture> metalfx_prepare(id<MTLDevice> dev, id<MTLTexture> in, double ow_d, double oh_d) {
    static int supported = -1;
    if (supported < 0) {
        supported = [MTLFXSpatialScalerDescriptor supportsDevice:dev] ? 1 : 0;
        METALFX_LOG("[metalfx] spatial scaler %s\n", supported ? "available" : "not supported on this GPU");
    }
    if (!supported || !in) return nil;
    NSUInteger ow = (NSUInteger)lround(ow_d), oh = (NSUInteger)lround(oh_d);
    if (ow <= in.width + 1 && oh <= in.height + 1) return nil;
    if (ow > 16384) ow = 16384;
    if (oh > 16384) oh = 16384;
    if (!g_fx || g_fx_iw != in.width || g_fx_ih != in.height || g_fx_ow != ow || g_fx_oh != oh || g_fx_fmt != in.pixelFormat) {
        MTLFXSpatialScalerDescriptor *d = [MTLFXSpatialScalerDescriptor new];
        d.inputWidth = in.width; d.inputHeight = in.height;
        d.outputWidth = ow; d.outputHeight = oh;
        d.colorTextureFormat = in.pixelFormat;
        d.outputTextureFormat = in.pixelFormat;
        d.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;  // (gamma-encoded 8-bit image)
        g_fx = [d newSpatialScalerWithDevice:dev];
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:in.pixelFormat width:ow height:oh mipmapped:NO];
        td.usage = g_fx ? g_fx.outputTextureUsage | MTLTextureUsageShaderRead : MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        g_fx_out = g_fx ? [dev newTextureWithDescriptor:td] : nil;
        g_fx_iw = in.width; g_fx_ih = in.height; g_fx_ow = ow; g_fx_oh = oh; g_fx_fmt = in.pixelFormat;
        g_fx_last = nil;
        METALFX_LOG("[metalfx] %lux%lu -> %lux%lu (%s)\n", (unsigned long)in.width, (unsigned long)in.height, (unsigned long)ow,
                    (unsigned long)oh, g_fx && g_fx_out ? "ok" : "FAILED");
    }
    return g_fx && g_fx_out ? g_fx_out : nil;
}

// Encodes the upscale of `in` (frame `frame`) into cb and returns the upscaled image. A repeated present of the same frame (the
// display runs faster than the game) reuses the image already in the output instead of scaling it again.
static id<MTLTexture> metalfx_encode(id<MTLCommandBuffer> cb, id<MTLTexture> in, uint64_t frame) {
    id<MTLTexture> base = in.parentTexture ? in.parentTexture : in;   // `in` is a fresh format view of the swapchain image each time
    if (g_fx_last && g_fx_last == base && g_fx_last_frame == frame) return g_fx_out;
    g_fx.colorTexture = in;
    g_fx.inputContentWidth = in.width;
    g_fx.inputContentHeight = in.height;
    g_fx.outputTexture = g_fx_out;
    [g_fx encodeToCommandBuffer:cb];
    g_fx_last = base;
    g_fx_last_frame = frame;
    return g_fx_out;
}

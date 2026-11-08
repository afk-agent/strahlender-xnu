// MetalFX temporal upscaling of the game's image before the present pass (metal.m present_gpu_locked), setting "upscaler" = temporal.
// The renderer jitters the 3D scene and records its motion and depth (taa.inc); here they are brought into the presented image's
// orientation, a reactive mask marks what was drawn over the scene (HUD, text, menus), and Apple's temporal scaler accumulates the
// jittered frames into an image of the screen's size - or of the image's own size when the screen is not larger (anti-aliasing only).
// Include once per program, after renderer.h (TaaFrame); define METALFX_LOG first to route its log lines.
#pragma once
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef METALFX_LOG
#include <stdio.h>
#define METALFX_LOG(...) fprintf(stderr, __VA_ARGS__)
#endif

static id<MTLFXTemporalScaler> g_tfx;
static id<MTLTexture> g_tfx_out, g_tfx_mv, g_tfx_depth, g_tfx_react;
static NSUInteger g_tfx_iw, g_tfx_ih, g_tfx_ow, g_tfx_oh;
static MTLPixelFormat g_tfx_fmt;
static bool g_tfx_has_react;
static id<MTLComputePipelineState> g_tfx_prep, g_tfx_prep_ms;   // (the second reads a multisampled depth target: sample 0)
static id<MTLTexture> g_tfx_dummy_mv, g_tfx_dummy_depth, g_tfx_dummy_color;
static uint64_t g_tfx_frame;   // frame whose image g_tfx_out holds (0: none)
static int g_tfx_logs;

// Motion, depth (the scene's target, upside down relative to the presented image when flip is set: Unity draws render textures that
// way and its final blit turns them) and the reactive mask (difference between the presented image and its copy from before the HUD).
static const char *g_tfx_prep_src =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "struct P { uint w, h, flip, has_mv, has_depth, has_copy; float far_depth; };\n"
    "#define DEPTH_T depth2d<float, access::read>\n#define DEPTH_READ(s) dep.read(s)\n#define PREP prep\n#include \"BODY\"\n";
static const char *g_tfx_prep_body =
    "kernel void PREP(texture2d<half, access::read> mv [[texture(0)]], DEPTH_T dep [[texture(1)]],\n"
    "                 texture2d<float, access::read> img [[texture(2)]], texture2d<float, access::read> cpy [[texture(3)]],\n"
    "                 texture2d<half, access::write> mvo [[texture(4)]], texture2d<float, access::write> dpo [[texture(5)]],\n"
    "                 texture2d<half, access::write> rmo [[texture(6)]], constant P &p [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {\n"
    "  if (g.x >= p.w || g.y >= p.h) return;\n"
    "  uint2 s = uint2(g.x, p.flip ? p.h - 1 - g.y : g.y);\n"
    "  half2 m = p.has_mv ? mv.read(s).xy : half2(0.0h);\n"
    "  if (p.flip) m.y = -m.y;\n"
    "  mvo.write(half4(m, 0.0h, 0.0h), g);\n"
    "  dpo.write(float4(p.has_depth ? DEPTH_READ(s) : p.far_depth), g);\n"
    "  if (p.has_copy) {\n"
    "    float3 d = abs(img.read(g).rgb - cpy.read(g).rgb);\n"
    "    rmo.write(half4(half(saturate((max(d.r, max(d.g, d.b)) - 1.5f / 255.0f) * 32.0f))), g);\n"
    "  } else rmo.write(half4(0.0h), g);\n"
    "}\n";

static id<MTLTexture> tfx_texture(id<MTLDevice> dev, MTLPixelFormat pf, NSUInteger w, NSUInteger h, MTLTextureUsage usage) {
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf width:w height:h mipmapped:NO];
    td.usage = usage;
    td.storageMode = MTLStorageModePrivate;
    return [dev newTextureWithDescriptor:td];
}

static bool metalfx_temporal_supported(id<MTLDevice> dev) {
    static int supported = -1;
    if (supported < 0) {
        supported = [MTLFXTemporalScalerDescriptor supportsDevice:dev] ? 1 : 0;
        METALFX_LOG("[metalfx] temporal scaler %s\n", supported ? "available" : "not supported on this GPU");
    }
    return supported == 1;
}

// Sets the scaler up for the image `in` shown at ow_d x oh_d pixels: the output is that size, at least the image's own (the temporal
// scaler does not reduce) and at most the device's largest scale. False when the device or the configuration is refused.
static bool metalfx_temporal_prepare(id<MTLDevice> dev, id<MTLTexture> in, double ow_d, double oh_d, float *scale) {
    if (!in || !metalfx_temporal_supported(dev)) return false;
    float maxs = [MTLFXTemporalScalerDescriptor supportedInputContentMaxScaleForDevice:dev];
    if (!(maxs >= 1)) maxs = 2;
    double s = fmin(ow_d / (double)in.width, oh_d / (double)in.height);
    if (!(s > 1.0)) s = 1.0;
    if (s > maxs) s = maxs;
    NSUInteger ow = (NSUInteger)lround(in.width * s), oh = (NSUInteger)lround(in.height * s);
    if (ow > 16384 || oh > 16384) { ow = in.width; oh = in.height; }
    *scale = (float)((double)ow / (double)in.width);
    if (g_tfx_iw == in.width && g_tfx_ih == in.height && g_tfx_ow == ow && g_tfx_oh == oh && g_tfx_fmt == in.pixelFormat) return g_tfx && g_tfx_out;
    MTLFXTemporalScalerDescriptor *d = [MTLFXTemporalScalerDescriptor new];
    d.inputWidth = in.width; d.inputHeight = in.height;
    d.outputWidth = ow; d.outputHeight = oh;
    d.colorTextureFormat = in.pixelFormat;
    d.depthTextureFormat = MTLPixelFormatR32Float;
    d.motionTextureFormat = MTLPixelFormatRG16Float;
    d.outputTextureFormat = in.pixelFormat;
    g_tfx_has_react = false;
    if (@available(macOS 14.4, iOS 17.4, *)) {
        d.reactiveMaskTextureEnabled = YES;
        d.reactiveMaskTextureFormat = MTLPixelFormatR8Unorm;
        g_tfx_has_react = true;
    }
    g_tfx = [d newTemporalScalerWithDevice:dev];
    g_tfx_out = g_tfx ? tfx_texture(dev, in.pixelFormat, ow, oh, g_tfx.outputTextureUsage | MTLTextureUsageShaderRead) : nil;
    MTLTextureUsage rw = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    g_tfx_mv = tfx_texture(dev, MTLPixelFormatRG16Float, in.width, in.height, rw);
    g_tfx_depth = tfx_texture(dev, MTLPixelFormatR32Float, in.width, in.height, rw);
    g_tfx_react = tfx_texture(dev, MTLPixelFormatR8Unorm, in.width, in.height, rw);
    if (!g_tfx_prep) {
        NSError *err = nil;
        NSString *body = @(g_tfx_prep_body);
        NSString *src = [@(g_tfx_prep_src) stringByReplacingOccurrencesOfString:@"#include \"BODY\"\n" withString:body];
        NSString *src_ms = [[[[src stringByReplacingOccurrencesOfString:@"depth2d<float, access::read>" withString:@"depth2d_ms<float, access::read>"]
            stringByReplacingOccurrencesOfString:@"dep.read(s)" withString:@"dep.read(s, 0)"]
            stringByReplacingOccurrencesOfString:@"#define PREP prep" withString:@"#define PREP prep_ms"] copy];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        id<MTLFunction> fn = [lib newFunctionWithName:@"prep"];
        g_tfx_prep = fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
        id<MTLLibrary> lib_ms = g_tfx_prep ? [dev newLibraryWithSource:src_ms options:nil error:&err] : nil;
        id<MTLFunction> fn_ms = [lib_ms newFunctionWithName:@"prep_ms"];
        g_tfx_prep_ms = fn_ms ? [dev newComputePipelineStateWithFunction:fn_ms error:&err] : nil;
        if (!g_tfx_prep || !g_tfx_prep_ms) {
            METALFX_LOG("[metalfx] temporal: preparation kernel failed: %s\n", err.localizedDescription.UTF8String);
            g_tfx_prep = nil;
        }
        g_tfx_dummy_mv = tfx_texture(dev, MTLPixelFormatRG16Float, 1, 1, MTLTextureUsageShaderRead);
        g_tfx_dummy_depth = tfx_texture(dev, MTLPixelFormatDepth32Float, 1, 1, MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
        g_tfx_dummy_color = tfx_texture(dev, MTLPixelFormatRGBA8Unorm, 1, 1, MTLTextureUsageShaderRead);
    }
    g_tfx_iw = in.width; g_tfx_ih = in.height; g_tfx_ow = ow; g_tfx_oh = oh; g_tfx_fmt = in.pixelFormat;
    g_tfx_frame = 0;
    bool ok = g_tfx && g_tfx_out && g_tfx_mv && g_tfx_depth && g_tfx_react && g_tfx_prep;
    if (g_tfx_logs++ < 40 || !ok)
        METALFX_LOG("[metalfx] temporal %lux%lu -> %lux%lu (x%.2f, reactive mask %s): %s\n", (unsigned long)in.width, (unsigned long)in.height,
                    (unsigned long)ow, (unsigned long)oh, *scale, g_tfx_has_react ? "yes" : "no", ok ? "ok" : "FAILED");
    if (!ok) g_tfx = nil;
    return ok;
}

// Encodes the temporal upscale of the presented image `in` (a raw format view) for frame `frame` and returns the output. tf: the
// frame's jitter, motion, depth and pre-HUD copy (`copy` is that copy as a view of in's format, nil without one); NULL for a repeated
// present, which reuses the image of the same frame (nil when there is none). flip: the scene's target is upside down relative to `in`.
static id<MTLTexture> metalfx_temporal_encode(id<MTLCommandBuffer> cb, id<MTLTexture> in, id<MTLTexture> copy, const TaaFrame *tf,
                                              bool flip, uint64_t frame) {
    if (!tf) return g_tfx_frame == frame && g_tfx ? g_tfx_out : nil;
    if (!g_tfx || in.width != g_tfx_iw || in.height != g_tfx_ih) return nil;
    bool mv_ok = tf->motion && tf->motion.width == in.width && tf->motion.height == in.height;
    bool depth_ok = mv_ok && tf->depth && tf->depth.width == in.width && tf->depth.height == in.height;
    bool copy_ok = copy && copy.width == in.width && copy.height == in.height;
    struct { uint32_t w, h, flip, has_mv, has_depth, has_copy; float far_depth; } p = {
        (uint32_t)in.width, (uint32_t)in.height, flip, mv_ok, depth_ok, copy_ok && g_tfx_has_react, tf->depth_reversed ? 0.0f : 1.0f};
    bool ms = depth_ok && tf->depth.textureType == MTLTextureType2DMultisample;
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    ce.label = @"metalfx temporal inputs";
    [ce setComputePipelineState:ms ? g_tfx_prep_ms : g_tfx_prep];
    [ce setTexture:mv_ok ? tf->motion : g_tfx_dummy_mv atIndex:0];
    [ce setTexture:depth_ok ? tf->depth : g_tfx_dummy_depth atIndex:1];
    [ce setTexture:in atIndex:2];
    [ce setTexture:copy_ok ? copy : g_tfx_dummy_color atIndex:3];
    [ce setTexture:g_tfx_mv atIndex:4];
    [ce setTexture:g_tfx_depth atIndex:5];
    [ce setTexture:g_tfx_react atIndex:6];
    [ce setBytes:&p length:sizeof p atIndex:0];
    [ce dispatchThreads:MTLSizeMake(in.width, in.height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [ce endEncoding];
    g_tfx.colorTexture = in;
    g_tfx.depthTexture = g_tfx_depth;
    g_tfx.motionTexture = g_tfx_mv;
    if (g_tfx_has_react) { if (@available(macOS 14.4, iOS 17.4, *)) g_tfx.reactiveMaskTexture = g_tfx_react; }
    g_tfx.outputTexture = g_tfx_out;
    g_tfx.inputContentWidth = in.width;
    g_tfx.inputContentHeight = in.height;
    // jitter: the content's shift in pixels (measured convention, runtime/test/metalfx_temporal_conv_test.m), in `in`'s orientation
    // (checked in the game, 2026-11-08: either sign reversed costs 2-3 dB against a native 1440p image, both 6 dB)
    g_tfx.jitterOffsetX = tf->jitter_x;
    g_tfx.jitterOffsetY = flip ? -tf->jitter_y : tf->jitter_y;
    g_tfx.motionVectorScaleX = (float)in.width;   // motion in uv units, previous - current
    g_tfx.motionVectorScaleY = (float)in.height;
    g_tfx.depthReversed = tf->depth_reversed;
    // history is dropped for a new scaler, after frames that were not accumulated, and for a frame without the scene's motion (no
    // 3D scene, or one the renderer could not record): its history would be reprojected as if nothing moved
    g_tfx.reset = g_tfx_frame == 0 || frame != g_tfx_frame + 1 || !mv_ok;
    [g_tfx encodeToCommandBuffer:cb];
    g_tfx_frame = frame;
    return g_tfx_out;
}

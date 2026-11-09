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
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef METALFX_LOG
#include <stdio.h>
#define METALFX_LOG(...) fprintf(stderr, __VA_ARGS__)
#endif

static id<MTLFXTemporalScaler> g_tfx;
static id<MTLTexture> g_tfx_out[2];   // two outputs, alternating: the previous frame's stays for frame generation (metalfx_interp.h)
static int g_tfx_oi;
static NSUInteger g_tfx_iw, g_tfx_ih, g_tfx_ow, g_tfx_oh;
static MTLPixelFormat g_tfx_fmt;
static bool g_tfx_has_react;
static uint64_t g_tfx_frame;   // frame whose image g_tfx_out[g_tfx_oi] holds (0: none)
static int g_tfx_logs;
// Inputs at the presented image's size, shared by the temporal scaler and the frame interpolator
static id<MTLComputePipelineState> g_tfx_prep, g_tfx_prep_ms;   // (the second reads a multisampled depth target: sample 0)
static id<MTLTexture> g_tfx_mv, g_tfx_mv_fg, g_tfx_depth, g_tfx_react;
static NSUInteger g_tfx_pw, g_tfx_ph;
static id<MTLTexture> g_tfx_dummy_mv, g_tfx_dummy_depth, g_tfx_dummy_color;

// Motion, depth (the scene's target, upside down relative to the presented image when flip is set: Unity draws render textures that
// way and its final blit turns them), the reactive mask (difference between the presented image and its copy from before the HUD) and
// the motion for frame generation (zero where the HUD is: the interpolated picture keeps the HUD in place).
static const char *g_tfx_prep_src =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "struct P { uint w, h, flip, has_mv, has_depth, has_copy; float far_depth; };\n"
    "#define DEPTH_T depth2d<float, access::read>\n#define DEPTH_READ(s) dep.read(s)\n#define PREP prep\n#include \"BODY\"\n";
static const char *g_tfx_prep_body =
    "kernel void PREP(texture2d<half, access::read> mv [[texture(0)]], DEPTH_T dep [[texture(1)]],\n"
    "                 texture2d<float, access::read> img [[texture(2)]], texture2d<float, access::read> cpy [[texture(3)]],\n"
    "                 texture2d<half, access::write> mvo [[texture(4)]], texture2d<float, access::write> dpo [[texture(5)]],\n"
    "                 texture2d<half, access::write> rmo [[texture(6)]], texture2d<half, access::write> mvf [[texture(7)]],\n"
    "                 constant P &p [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {\n"
    "  if (g.x >= p.w || g.y >= p.h) return;\n"
    "  uint2 s = uint2(g.x, p.flip ? p.h - 1 - g.y : g.y);\n"
    "  half2 m = p.has_mv ? mv.read(s).xy : half2(0.0h);\n"
    "  if (p.flip) m.y = -m.y;\n"
    "  mvo.write(half4(m, 0.0h, 0.0h), g);\n"
    "  dpo.write(float4(p.has_depth ? DEPTH_READ(s) : p.far_depth), g);\n"
    "  half r = 0.0h;\n"
    "  if (p.has_copy) {\n"
    "    float3 d = abs(img.read(g).rgb - cpy.read(g).rgb);\n"
    "    r = half(saturate((max(d.r, max(d.g, d.b)) - 1.5f / 255.0f) * 32.0f));\n"
    "  }\n"
    "  rmo.write(half4(r), g);\n"
    "  mvf.write(half4(m * (1.0h - r), 0.0h, 0.0h), g);\n"
    "}\n";

// Memory check before MetalFX allocates (measured on an M1: the temporal scaler takes about 52 bytes per output pixel, 163 MB for 720p to
// an 11" iPad's 2360x1328, the frame interpolator about 57 bytes per input pixel, 52 MB at 720p). On iPhone/iPad the app's memory limit is
// close to the game's own use (issue #57): with less than `need` plus a 384 MB margin left, the effect is not set up and the cheaper path
// stays (logged). Elsewhere it always fits.
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif
static bool metalfx_memory_ok(const char *what, uint64_t need) {
#if TARGET_OS_IPHONE
    uint64_t avail = os_proc_available_memory();
    if (avail && avail < need + (384ull << 20)) {
        static int n;
        if (n++ < 10) METALFX_LOG("[metalfx] %s not set up: it needs about %llu MB, the app has %llu MB left before its limit\n", what,
                                  (unsigned long long)(need >> 20), (unsigned long long)(avail >> 20));
        return false;
    }
#else
    (void)what; (void)need;
#endif
    return true;
}

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

// Encodes the preparation of the frame's inputs for the presented image `in` (a raw format view): tf has the scene's motion, depth and
// the pre-HUD copy (`copy`, as a view of in's format; nil without one). False when the kernel could not be built. *motion: the frame
// had the scene's motion.
static bool tfx_prep_encode(id<MTLDevice> dev, id<MTLCommandBuffer> cb, id<MTLTexture> in, id<MTLTexture> copy, const TaaFrame *tf,
                            bool flip, bool *motion) {
    if (!g_tfx_prep) {
        static bool failed;
        if (failed) return false;
        NSError *err = nil;
        NSString *src = [@(g_tfx_prep_src) stringByReplacingOccurrencesOfString:@"#include \"BODY\"\n" withString:@(g_tfx_prep_body)];
        NSString *src_ms = [[[src stringByReplacingOccurrencesOfString:@"depth2d<float, access::read>" withString:@"depth2d_ms<float, access::read>"]
            stringByReplacingOccurrencesOfString:@"dep.read(s)" withString:@"dep.read(s, 0)"]
            stringByReplacingOccurrencesOfString:@"#define PREP prep" withString:@"#define PREP prep_ms"];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        id<MTLFunction> fn = [lib newFunctionWithName:@"prep"];
        id<MTLComputePipelineState> p = fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
        id<MTLLibrary> lib_ms = p ? [dev newLibraryWithSource:src_ms options:nil error:&err] : nil;
        id<MTLFunction> fn_ms = [lib_ms newFunctionWithName:@"prep_ms"];
        g_tfx_prep_ms = fn_ms ? [dev newComputePipelineStateWithFunction:fn_ms error:&err] : nil;
        if (!p || !g_tfx_prep_ms) {
            METALFX_LOG("[metalfx] temporal: preparation kernel failed: %s\n", err.localizedDescription.UTF8String);
            failed = true;
            return false;
        }
        g_tfx_prep = p;
        g_tfx_dummy_mv = tfx_texture(dev, MTLPixelFormatRG16Float, 1, 1, MTLTextureUsageShaderRead);
        g_tfx_dummy_depth = tfx_texture(dev, MTLPixelFormatDepth32Float, 1, 1, MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
        g_tfx_dummy_color = tfx_texture(dev, MTLPixelFormatRGBA8Unorm, 1, 1, MTLTextureUsageShaderRead);
    }
    if (g_tfx_pw != in.width || g_tfx_ph != in.height) {
        MTLTextureUsage rw = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        g_tfx_mv = tfx_texture(dev, MTLPixelFormatRG16Float, in.width, in.height, rw);
        g_tfx_mv_fg = tfx_texture(dev, MTLPixelFormatRG16Float, in.width, in.height, rw);
        g_tfx_depth = tfx_texture(dev, MTLPixelFormatR32Float, in.width, in.height, rw);
        g_tfx_react = tfx_texture(dev, MTLPixelFormatR8Unorm, in.width, in.height, rw);
        g_tfx_pw = in.width; g_tfx_ph = in.height;
    }
    bool mv_ok = tf->motion && tf->motion.width == in.width && tf->motion.height == in.height;
    bool depth_ok = mv_ok && tf->depth && tf->depth.width == in.width && tf->depth.height == in.height;
    bool copy_ok = copy && copy.width == in.width && copy.height == in.height;
    struct { uint32_t w, h, flip, has_mv, has_depth, has_copy; float far_depth; } p = {
        (uint32_t)in.width, (uint32_t)in.height, flip, mv_ok, depth_ok, copy_ok, tf->depth_reversed ? 0.0f : 1.0f};
    bool ms = depth_ok && tf->depth.textureType == MTLTextureType2DMultisample;
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    ce.label = @"metalfx inputs";
    [ce setComputePipelineState:ms ? g_tfx_prep_ms : g_tfx_prep];
    [ce setTexture:mv_ok ? tf->motion : g_tfx_dummy_mv atIndex:0];
    [ce setTexture:depth_ok ? tf->depth : g_tfx_dummy_depth atIndex:1];
    [ce setTexture:in atIndex:2];
    [ce setTexture:copy_ok ? copy : g_tfx_dummy_color atIndex:3];
    [ce setTexture:g_tfx_mv atIndex:4];
    [ce setTexture:g_tfx_depth atIndex:5];
    [ce setTexture:g_tfx_react atIndex:6];
    [ce setTexture:g_tfx_mv_fg atIndex:7];
    [ce setBytes:&p length:sizeof p atIndex:0];
    [ce dispatchThreads:MTLSizeMake(in.width, in.height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [ce endEncoding];
    *motion = mv_ok;
    return true;
}

// Sets the scaler up for the image `in` shown at ow_d x oh_d pixels: the output is that size, at least the image's own (the temporal
// scaler does not reduce), at most `cap` times it (setting "upscale_max", 0: no cap) and the device's largest scale. False when the
// device or the configuration is refused.
static bool metalfx_temporal_prepare(id<MTLDevice> dev, id<MTLTexture> in, double ow_d, double oh_d, float cap, float *scale) {
    if (!in || !metalfx_temporal_supported(dev)) return false;
    float maxs = [MTLFXTemporalScalerDescriptor supportedInputContentMaxScaleForDevice:dev];
    if (!(maxs >= 1)) maxs = 2;
    if (cap >= 1 && cap < maxs) maxs = cap;
    double s = fmin(ow_d / (double)in.width, oh_d / (double)in.height);
    if (!(s > 1.0)) s = 1.0;
    if (s > maxs) s = maxs;
    NSUInteger ow = (NSUInteger)lround(in.width * s), oh = (NSUInteger)lround(in.height * s);
    if (ow > 16384 || oh > 16384) { ow = in.width; oh = in.height; }
    *scale = (float)((double)ow / (double)in.width);
    if (g_tfx_iw == in.width && g_tfx_ih == in.height && g_tfx_ow == ow && g_tfx_oh == oh && g_tfx_fmt == in.pixelFormat) return g_tfx != nil;
    g_tfx = nil; g_tfx_out[0] = g_tfx_out[1] = nil;   // (the old one goes before the new one is made)
    g_tfx_iw = in.width; g_tfx_ih = in.height; g_tfx_ow = ow; g_tfx_oh = oh; g_tfx_fmt = in.pixelFormat;   // (a refusal is not retried)
    g_tfx_frame = 0;
    if (!metalfx_memory_ok("temporal scaler", (uint64_t)ow * oh * 60 + (uint64_t)in.width * in.height * 16)) return false;
    size_t mem0 = dev.currentAllocatedSize;
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
    for (int i = 0; i < 2; i++)
        g_tfx_out[i] = g_tfx ? tfx_texture(dev, in.pixelFormat, ow, oh, g_tfx.outputTextureUsage | MTLTextureUsageShaderRead) : nil;
    bool ok = g_tfx && g_tfx_out[0] && g_tfx_out[1];
    if (g_tfx_logs++ < 40 || !ok)
        METALFX_LOG("[metalfx] temporal %lux%lu -> %lux%lu (x%.2f, reactive mask %s): %s, %.0f MB\n", (unsigned long)in.width, (unsigned long)in.height,
                    (unsigned long)ow, (unsigned long)oh, *scale, g_tfx_has_react ? "yes" : "no", ok ? "ok" : "FAILED",
                    (double)((int64_t)dev.currentAllocatedSize - (int64_t)mem0) / 1048576.0);
    if (!ok) g_tfx = nil;
    return ok;
}

// Encodes the temporal upscale of the presented image `in` (a raw format view) for frame `frame` and returns the output; the inputs
// (tfx_prep_encode) are encoded before it. tf: the frame's jitter; motion: the frame had the scene's motion. tf NULL: a repeated present,
// which reuses the image of the same frame (nil when there is none). flip: the scene's target is upside down relative to `in`.
static id<MTLTexture> metalfx_temporal_encode(id<MTLCommandBuffer> cb, id<MTLTexture> in, const TaaFrame *tf, bool motion, bool flip,
                                              uint64_t frame) {
    if (!tf) return g_tfx_frame == frame && g_tfx ? g_tfx_out[g_tfx_oi] : nil;
    if (!g_tfx || in.width != g_tfx_iw || in.height != g_tfx_ih || g_tfx_pw != in.width || g_tfx_ph != in.height) return nil;
    g_tfx_oi ^= 1;
    g_tfx.colorTexture = in;
    g_tfx.depthTexture = g_tfx_depth;
    g_tfx.motionTexture = g_tfx_mv;
    if (g_tfx_has_react) { if (@available(macOS 14.4, iOS 17.4, *)) g_tfx.reactiveMaskTexture = g_tfx_react; }
    g_tfx.outputTexture = g_tfx_out[g_tfx_oi];
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
    g_tfx.reset = g_tfx_frame == 0 || frame != g_tfx_frame + 1 || !motion;
    [g_tfx encodeToCommandBuffer:cb];
    g_tfx_frame = frame;
    return g_tfx_out[g_tfx_oi];
}

// The setting went off: the scaler and its outputs are released (about 50 bytes per output pixel); with frame generation off too, the
// prepared inputs as well.
static void metalfx_temporal_release(bool inputs_too) {
    if (g_tfx || g_tfx_out[0]) METALFX_LOG("[metalfx] temporal scaler released\n");
    g_tfx = nil; g_tfx_out[0] = g_tfx_out[1] = nil;
    g_tfx_iw = g_tfx_ih = g_tfx_ow = g_tfx_oh = 0;
    g_tfx_frame = 0;
    if (inputs_too) { g_tfx_mv = g_tfx_mv_fg = g_tfx_depth = g_tfx_react = nil; g_tfx_pw = g_tfx_ph = 0; }
}

// Orientation of the scene's targets relative to the presented image. Unity draws render textures upside down on this platform and its
// final blit turns them; that is the starting assumption. Every 30th frame with motion, the brightness of the scene's colour target and of
// the presented image is sampled on a 48 x 27 grid (4 x 4 taps each) and correlated both ways; a clearly better match (correlation above
// 0.5 and 0.15 over the other) moves a score, and the score decides. Post-processing keeps the order of brightness, a menu over a blurred
// scene matches neither way and changes nothing. BDSP_TAA_FLIP=0/1 fixes it.
#define TFX_OGX 48
#define TFX_OGY 27
static id<MTLComputePipelineState> g_tfx_orient;
static _Atomic int g_tfx_flip_score = 3;
static const char *g_tfx_orient_src =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "static float lum(float3 v) { float l = dot(v, float3(0.2126f, 0.7152f, 0.0722f)); return isfinite(l) ? min(l, 16.0f) : 0.0f; }\n"   // (HDR targets hold NaN/Inf pixels)
    "kernel void orient(texture2d<float, access::sample> sc [[texture(0)]], texture2d<float, access::sample> im [[texture(1)]],\n"
    "                   device float *o [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {\n"
    "  if (g.x >= 48u || g.y >= 27u) return;\n"
    "  constexpr sampler s(filter::linear, address::clamp_to_edge);\n"
    "  float a = 0, b = 0, c = 0;\n"
    "  for (uint j = 0; j < 4; j++) for (uint i = 0; i < 4; i++) {\n"
    "    float2 uv = (float2(g) + (float2(i, j) + 0.5f) / 4.0f) / float2(48.0f, 27.0f);\n"
    "    a += lum(sc.sample(s, uv).rgb); b += lum(sc.sample(s, float2(uv.x, 1.0f - uv.y)).rgb); c += lum(im.sample(s, uv).rgb);\n"
    "  }\n"
    "  uint k = (g.y * 48u + g.x) * 3u;\n"
    "  o[k] = a; o[k + 1] = b; o[k + 2] = c;\n"
    "}\n";

static double tfx_ncc(const float *v, int off) {   // correlation of column off (0 or 1) with column 2
    double sa = 0, sc = 0, saa = 0, scc = 0, sac = 0;
    const int n = TFX_OGX * TFX_OGY;
    for (int i = 0; i < n; i++) { double a = v[i * 3 + off], c = v[i * 3 + 2]; sa += a; sc += c; saa += a * a; scc += c * c; sac += a * c; }
    double va = saa - sa * sa / n, vc = scc - sc * sc / n;
    return va > 1e-9 && vc > 1e-9 ? (sac - sa * sc / n) / sqrt(va * vc) : 0;
}

static bool tfx_scene_flipped(void) {
    static int forced = -2;
    if (forced == -2) { const char *e = getenv("BDSP_TAA_FLIP"); forced = e ? atoi(e) != 0 : -1; }
    return forced >= 0 ? forced : atomic_load(&g_tfx_flip_score) > 0;
}

static void tfx_orient_check(id<MTLDevice> dev, id<MTLCommandBuffer> cb, id<MTLTexture> scene, id<MTLTexture> in, uint64_t frame) {
    if (frame % 30 || !scene || scene.textureType != MTLTextureType2D || scene.width != in.width || scene.height != in.height) return;
    if (!g_tfx_orient) {
        static bool failed;
        if (failed) return;
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(g_tfx_orient_src) options:nil error:&err];
        id<MTLFunction> fn = [lib newFunctionWithName:@"orient"];
        g_tfx_orient = fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
        if (!g_tfx_orient) { failed = true; METALFX_LOG("[metalfx] orientation check unavailable: %s\n", err.localizedDescription.UTF8String); return; }
    }
    id<MTLBuffer> buf = [dev newBufferWithLength:TFX_OGX * TFX_OGY * 3 * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    ce.label = @"metalfx orientation check";
    [ce setComputePipelineState:g_tfx_orient];
    [ce setTexture:scene atIndex:0];
    [ce setTexture:in atIndex:1];
    [ce setBuffer:buf offset:0 atIndex:0];
    [ce dispatchThreads:MTLSizeMake(TFX_OGX, TFX_OGY, 1) threadsPerThreadgroup:MTLSizeMake(16, 9, 1)];
    [ce endEncoding];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        (void)done;
        double direct = tfx_ncc(buf.contents, 0), turned = tfx_ncc(buf.contents, 1);
        int before = atomic_load(&g_tfx_flip_score), s = before;
        if (turned > 0.5 && turned > direct + 0.15 && s < 5) s++;
        else if (direct > 0.5 && direct > turned + 0.15 && s > -5) s--;
        atomic_store(&g_tfx_flip_score, s);
        if ((before > 0) != (s > 0)) METALFX_LOG("[metalfx] scene orientation: %s (correlation as drawn %.2f, turned %.2f)\n", s > 0 ? "upside down" : "upright", direct, turned);
        const char *lg = getenv("BDSP_TAA_LOG");
        if (lg && (frame % 600 == 0 || atoi(lg) >= 2))
            METALFX_LOG("[metalfx] orientation check: as drawn %.2f, turned %.2f, score %d\n", direct, turned, s);
    }];
}

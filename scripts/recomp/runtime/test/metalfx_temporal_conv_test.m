// Measured 2026-11-08 (Apple silicon, macOS 27): jitterOffset = -(sample offset) wins by ~6 dB on a still scene, motion vectors
// (previous - current, y down) by ~2 dB over the opposite sign and over zero motion at 1-16 output px per frame.
// Measures the conventions of Apple's MetalFX temporal scaler that the renderer relies on (metalfx_temporal.h): the sign of the
// jitter offset and of the motion vectors. A synthetic scene (analytic, so any sub-pixel position can be sampled exactly) is
// rendered at half the output size with a per-frame sub-pixel jitter, optionally scrolling, and upscaled by the temporal scaler for
// a number of frames; the output is compared with the scene sampled at the output size. The convention that matches gives a much
// higher PSNR than its opposite.
// Usage: metalfx_temporal_conv_test   (prints one line per variant; exit 1 if the expected conventions do not win)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define OW 1280
#define OH 720
#define IW 640
#define IH 360

// The scene at output pixel coordinates (x, y) (continuous); shifted by `scroll` output pixels to the right and down.
static void scene(double x, double y, double scroll, float out[4]) {
    x -= scroll; y -= scroll * 0.5;
    double r = 0.35 + 0.25 * sin(x * 0.031) + 0.25 * ((int)floor((x + 2 * y) / 23.0) & 1);
    double g = 0.4 + 0.3 * cos(y * 0.047);
    double b = 0.3 + 0.6 * (fmod(fabs(x * 0.7 - y * 0.3), 9.0) < 1.2);   // thin lines: aliasing shows at once
    double cx = x - 700, cy = y - 300, d = sqrt(cx * cx + cy * cy);
    if (fabs(d - 150) < 1.5) r = g = b = 1;
    out[0] = (float)fmin(r, 1); out[1] = (float)fmin(g, 1); out[2] = (float)fmin(b, 1); out[3] = 1;
}

static double halton(int i, int b) { double f = 1, r = 0; for (i++; i > 0; i /= b) { f /= b; r += f * (i % b); } return r; }

// jsign: sign of jitterOffset relative to the sample offset; msign: sign of the motion vectors relative to (previous - current)
static double run(id<MTLDevice> dev, id<MTLCommandQueue> q, int jsign, int msign, double speed, int frames) {
    MTLFXTemporalScalerDescriptor *d = [MTLFXTemporalScalerDescriptor new];
    d.colorTextureFormat = MTLPixelFormatRGBA32Float;
    d.depthTextureFormat = MTLPixelFormatDepth32Float;
    d.motionTextureFormat = MTLPixelFormatRG16Float;
    d.outputTextureFormat = MTLPixelFormatRGBA32Float;
    d.inputWidth = IW; d.inputHeight = IH; d.outputWidth = OW; d.outputHeight = OH;
    id<MTLFXTemporalScaler> s = [d newTemporalScalerWithDevice:dev];
    if (!s) { fprintf(stderr, "no temporal scaler\n"); exit(2); }
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:IW height:IH mipmapped:NO];
    td.usage = s.colorTextureUsage; td.storageMode = MTLStorageModeShared;
    id<MTLTexture> color = [dev newTextureWithDescriptor:td];
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:IW height:IH mipmapped:NO];
    td.usage = s.depthTextureUsage | MTLTextureUsageRenderTarget; td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> depth = [dev newTextureWithDescriptor:td];
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float width:IW height:IH mipmapped:NO];
    td.usage = s.motionTextureUsage; td.storageMode = MTLStorageModeShared;
    id<MTLTexture> motion = [dev newTextureWithDescriptor:td];
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:OW height:OH mipmapped:NO];
    td.usage = s.outputTextureUsage; td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> out = [dev newTextureWithDescriptor:td];
    id<MTLBuffer> rb = [dev newBufferWithLength:(NSUInteger)OW * OH * 16 options:MTLResourceStorageModeShared];
    float *px = malloc((size_t)IW * IH * 16);
    uint16_t *mv = malloc((size_t)IW * IH * 4);
    double total = 0; int counted = 0;
    for (int f = 0; f < frames; f++) {
        double jx = halton(f % 32, 2) - 0.5, jy = halton(f % 32, 3) - 0.5;   // input pixels; the sample sits at centre + j
        double scroll = speed * f;
        for (int y = 0; y < IH; y++) for (int x = 0; x < IW; x++)
            scene((x + 0.5 + jx) * OW / IW, (y + 0.5 + jy) * OH / IH, scroll, px + ((size_t)y * IW + x) * 4);
        [color replaceRegion:MTLRegionMake2D(0, 0, IW, IH) mipmapLevel:0 withBytes:px bytesPerRow:IW * 16];
        // previous - current, in input UV units: the content moved by +speed output px (x) and +speed/2 (y)
        __fp16 mx = (__fp16)(msign * -(speed / OW)), my = (__fp16)(msign * -(speed * 0.5 / OH));
        for (size_t i = 0; i < (size_t)IW * IH; i++) { ((__fp16 *)mv)[2 * i] = mx; ((__fp16 *)mv)[2 * i + 1] = my; }
        [motion replaceRegion:MTLRegionMake2D(0, 0, IW, IH) mipmapLevel:0 withBytes:mv bytesPerRow:IW * 4];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.depthAttachment.texture = depth; rp.depthAttachment.loadAction = MTLLoadActionClear;
        rp.depthAttachment.clearDepth = 0.5; rp.depthAttachment.storeAction = MTLStoreActionStore;
        [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
        s.colorTexture = color; s.depthTexture = depth; s.motionTexture = motion; s.outputTexture = out;
        s.inputContentWidth = IW; s.inputContentHeight = IH;
        s.jitterOffsetX = (float)(jsign * jx); s.jitterOffsetY = (float)(jsign * jy);
        s.motionVectorScaleX = IW; s.motionVectorScaleY = IH;
        s.reset = f == 0;
        [s encodeToCommandBuffer:cb];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl copyFromTexture:out sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(OW, OH, 1)
                   toBuffer:rb destinationOffset:0 destinationBytesPerRow:OW * 16 destinationBytesPerImage:(NSUInteger)OW * OH * 16];
        [bl endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (f >= frames - 8) {   // the last frames, after the history has filled
            float *ref = malloc((size_t)OW * OH * 16);
            for (int y = 0; y < OH; y++) for (int x = 0; x < OW; x++) scene(x + 0.5, y + 0.5, scroll, ref + ((size_t)y * OW + x) * 4);
            // a border of 16 px is left out (content entering the picture has no history)
            double se = 0; size_t n = 0;
            const float *o = rb.contents;
            for (int y = 16; y < OH - 16; y++) for (int x = 16; x < OW - 16; x++) for (int c = 0; c < 3; c++) {
                double dd = o[((size_t)y * OW + x) * 4 + c] - ref[((size_t)y * OW + x) * 4 + c]; se += dd * dd; n++;
            }
            total += 10 * log10(1 / fmax(se / n, 1e-12)); counted++;
            free(ref);
        }
    }
    free(px); free(mv);
    return total / counted;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (![MTLFXTemporalScalerDescriptor supportsDevice:dev]) { fprintf(stderr, "MetalFX temporal not supported\n"); return 2; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        double jp = run(dev, q, +1, +1, 0, 48), jn = run(dev, q, -1, +1, 0, 48);
        printf("static scene, jitterOffset = +sample offset: %.2f dB, = -sample offset: %.2f dB\n", jp, jn);
        int js = jp > jn ? +1 : -1;
        double mp = run(dev, q, js, +1, 4.0, 48), mn = run(dev, q, js, -1, 4.0, 48), m0 = run(dev, q, js, 0, 4.0, 48);
        printf("scrolling 4 px/frame, motion = previous - current: %.2f dB, = current - previous: %.2f dB, zero: %.2f dB\n", mp, mn, m0);
        bool ok = js == +1 ? jp > jn + 1 : jn > jp + 1;
        ok = ok && mp > mn + 1 && mp > m0 + 1;   // (measured on an M-series Mac: +2 dB at 1-16 px per frame)
        printf("conventions: jitterOffset = %s sample offset (input pixels), motion = previous - current (input UV with scale = input size): %s\n",
               js > 0 ? "+" : "-", ok ? "confirmed" : "NOT confirmed");
        return ok ? 0 : 1;
    }
}

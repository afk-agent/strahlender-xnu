// Measured 2026-11-08 (M1, macOS 27): with history (a sequence, not one pair) the output sits at the midpoint (25.5 dB against 20.9
// for repeating the older frame); field of view and planes make no difference; colour at twice the motion's size works (41.8 dB);
// 6 ms for a 1280x720 picture, 12 ms at 1920x1080, 19 ms at 2560x1440.
// Measures Apple's MetalFX frame interpolator (macOS/iOS 26) the way the renderer feeds it (metalfx_interp.h): two frames of a
// synthetic scene that scrolls, with the motion (previous - current, uv units) and depth of the newer one, and the frame halfway
// between them as the reference. Prints PSNR of the generated frame against that reference, against simply repeating the older
// or the newer frame, for camera parameters that are right and wrong, and with colour at the depth's size or at twice it. Also
// times it. Exit 1 when the interpolated frame is not clearly better than repeating a frame.
// Usage: metalfx_interp_test
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// scene at pixel (x, y) of a w x h image, content shifted by (sx, sy) pixels: a far textured backdrop and a near square that moves faster
static void scene(double x, double y, int w, int h, double t, float out[4], float *depth) {
    double bx = x - 6 * t * w / 1280.0, by = y - 2 * t * h / 720.0;       // backdrop: 6 px per frame at 1280 wide
    double r = 0.4 + 0.3 * sin(bx * 0.05), g = 0.4 + 0.3 * cos(by * 0.07), b = ((int)floor(bx / 23) + (int)floor(by / 19)) & 1 ? 0.8 : 0.2;
    double d = 0.05;                                                        // reversed depth: far
    double qx = w * 0.3 + 20 * t * w / 1280.0, qy = h * 0.4;                // square: 20 px per frame
    if (x > qx && x < qx + w * 0.15 && y > qy && y < qy + h * 0.25) { r = 0.9; g = 0.2 + 0.5 * (((int)(x - qx) / 8) & 1); b = 0.1; d = 0.6; }
    out[0] = (float)r; out[1] = (float)g; out[2] = (float)b; out[3] = 1;
    if (depth) *depth = (float)d;
}

static void motion_at(double x, double y, int w, int h, double t, __fp16 *m) {
    float c[4], d;
    scene(x, y, w, h, t, c, &d);
    double dx = d > 0.5 ? 20 : 6, dy = d > 0.5 ? 0 : 2;
    m[0] = (__fp16)(-dx * w / 1280.0 / w); m[1] = (__fp16)(-dy * h / 720.0 / h);
}
static id<MTLTexture> tex(id<MTLDevice> d, MTLPixelFormat f, int w, int h, MTLTextureUsage u, MTLStorageMode sm) {
    MTLTextureDescriptor *t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:h mipmapped:NO];
    t.usage = u; t.storageMode = sm;
    return [d newTextureWithDescriptor:t];
}

static double psnr(const float *a, const float *b, int w, int h) {
    double se = 0; size_t n = 0;
    for (int y = 8; y < h - 8; y++) for (int x = 8; x < w - 8; x++) for (int c = 0; c < 3; c++) {
        double e = a[((size_t)y * w + x) * 4 + c] - b[((size_t)y * w + x) * 4 + c]; se += e * e; n++;
    }
    return 10 * log10(1 / fmax(se / n, 1e-12));
}

static float *render(int w, int h, double t) {
    float *p = malloc((size_t)w * h * 16);
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) scene(x + 0.5, y + 0.5, w, h, t, p + ((size_t)y * w + x) * 4, NULL);
    return p;
}

// a sequence of frames t = 0, 1, 2, ...: each call gets frame k, frame k - 1 as the previous colour and frame k's motion and depth;
// returns PSNR of the last output against the frame halfway between the last two (and the time in ms)
static double run(id<MTLDevice> dev, id<MTLCommandQueue> q, int iw, int ih, int cw, int ch, float fov, float nearp, float farp, double *ms,
                  double *rep_prev, double *rep_cur) {
    MTLFXFrameInterpolatorDescriptor *d = [MTLFXFrameInterpolatorDescriptor new];
    d.colorTextureFormat = MTLPixelFormatRGBA16Float; d.outputTextureFormat = MTLPixelFormatRGBA16Float;
    d.depthTextureFormat = MTLPixelFormatR32Float; d.motionTextureFormat = MTLPixelFormatRG16Float;
    d.uiTextureFormat = MTLPixelFormatRGBA16Float;
    d.inputWidth = iw; d.inputHeight = ih; d.outputWidth = cw; d.outputHeight = ch;
    id<MTLFXFrameInterpolator> fi = [d newFrameInterpolatorWithDevice:dev];
    if (!fi) { fprintf(stderr, "interpolator refused %dx%d -> %dx%d\n", iw, ih, cw, ch); return -1; }
    MTLStorageMode sh = MTLStorageModeShared;
    const int N = 10;
    id<MTLTexture> col[N];
    for (int k = 0; k < N; k++) {
        float *p = render(cw, ch, k);
        __fp16 *hh = malloc((size_t)cw * ch * 8);
        for (size_t i = 0; i < (size_t)cw * ch * 4; i++) hh[i] = (__fp16)p[i];
        col[k] = tex(dev, MTLPixelFormatRGBA16Float, cw, ch, fi.colorTextureUsage, sh);
        [col[k] replaceRegion:MTLRegionMake2D(0, 0, cw, ch) mipmapLevel:0 withBytes:hh bytesPerRow:cw * 8];
        free(p); free(hh);
    }
    id<MTLTexture> dt[N], mt[N];
    float *dp = malloc((size_t)iw * ih * 4); __fp16 *mv = malloc((size_t)iw * ih * 4);
    for (int k = 0; k < N; k++) {
        for (int y = 0; y < ih; y++) for (int x = 0; x < iw; x++) {
            float c[4];
            scene(x + 0.5, y + 0.5, iw, ih, k, c, dp + (size_t)y * iw + x);
            motion_at(x + 0.5, y + 0.5, iw, ih, k, mv + ((size_t)y * iw + x) * 2);
        }
        dt[k] = tex(dev, MTLPixelFormatR32Float, iw, ih, fi.depthTextureUsage, sh); mt[k] = tex(dev, MTLPixelFormatRG16Float, iw, ih, fi.motionTextureUsage, sh);
        [dt[k] replaceRegion:MTLRegionMake2D(0, 0, iw, ih) mipmapLevel:0 withBytes:dp bytesPerRow:iw * 4];
        [mt[k] replaceRegion:MTLRegionMake2D(0, 0, iw, ih) mipmapLevel:0 withBytes:mv bytesPerRow:iw * 4];
    }
    id<MTLTexture> out = tex(dev, MTLPixelFormatRGBA16Float, cw, ch, fi.outputTextureUsage, MTLStorageModePrivate);
    id<MTLBuffer> rb = [dev newBufferWithLength:(size_t)cw * ch * 8 options:MTLResourceStorageModeShared];
    double tsum = 0; int tn = 0;
    for (int k = 1; k < N; k++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        fi.colorTexture = col[k]; fi.prevColorTexture = col[k - 1]; fi.depthTexture = dt[k]; fi.motionTexture = mt[k]; fi.outputTexture = out;
        fi.motionVectorScaleX = iw; fi.motionVectorScaleY = ih;
        fi.deltaTime = 1.0f / 30; fi.fieldOfView = fov; fi.aspectRatio = (float)cw / ch; fi.nearPlane = nearp; fi.farPlane = farp;
        fi.depthReversed = YES; fi.shouldResetHistory = k == 1;
        [fi encodeToCommandBuffer:cb];
        if (k == N - 1) {
            id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
            [bl copyFromTexture:out sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(cw, ch, 1) toBuffer:rb
                destinationOffset:0 destinationBytesPerRow:cw * 8 destinationBytesPerImage:(size_t)cw * ch * 8];
            [bl endEncoding];
        }
        [cb commit]; [cb waitUntilCompleted];
        if (k >= 4) { tsum += cb.GPUEndTime - cb.GPUStartTime; tn++; }
    }
    float *o = malloc((size_t)cw * ch * 16);
    const __fp16 *r = rb.contents;
    for (size_t i = 0; i < (size_t)cw * ch * 4; i++) o[i] = (float)r[i];
    float *mid = render(cw, ch, N - 1.5), *p0 = render(cw, ch, N - 2), *p1 = render(cw, ch, N - 1);
    double ps = psnr(o, mid, cw, ch);
    if (rep_prev) *rep_prev = psnr(p0, mid, cw, ch);
    if (rep_cur) *rep_cur = psnr(p1, mid, cw, ch);
    *ms = tsum / tn * 1000;
    free(p0); free(p1); free(mid); free(dp); free(mv); free(o);
    return ps;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (@available(macOS 26.0, *)) {
            if (![MTLFXFrameInterpolatorDescriptor supportsDevice:dev]) { printf("frame interpolation not supported on %s\n", dev.name.UTF8String); return 2; }
            id<MTLCommandQueue> q = [dev newCommandQueue];
            double ms, rp, rc;
            double a = run(dev, q, 640, 360, 640, 360, 30, 0.3f, 1000, &ms, &rp, &rc);
            printf("640x360: interpolated %.2f dB (repeat older %.2f, newer %.2f), %.2f ms\n", a, rp, rc, ms);
            double b = run(dev, q, 640, 360, 640, 360, 60, 0.1f, 5000, &ms, NULL, NULL);
            printf("640x360, other camera numbers (fov 60, near 0.1, far 5000): %.2f dB\n", b);
            double c = run(dev, q, 640, 360, 1280, 720, 30, 0.3f, 1000, &ms, &rp, &rc);
            printf("motion 640x360, colour 1280x720: %.2f dB (repeat %.2f / %.2f), %.2f ms\n", c, rp, rc, ms);
            run(dev, q, 1280, 720, 1280, 720, 30, 0.3f, 1000, &ms, NULL, NULL); printf("cost 1280x720: %.2f ms\n", ms);
            run(dev, q, 1280, 720, 2560, 1440, 30, 0.3f, 1000, &ms, NULL, NULL); printf("cost 1280x720 -> 2560x1440: %.2f ms\n", ms);
            run(dev, q, 1920, 1080, 1920, 1080, 30, 0.3f, 1000, &ms, NULL, NULL); printf("cost 1920x1080: %.2f ms\n", ms);
            return a > fmax(rp, rc) + 3 ? 0 : 1;
        }
        printf("needs macOS 26\n");
        return 2;
    }
}

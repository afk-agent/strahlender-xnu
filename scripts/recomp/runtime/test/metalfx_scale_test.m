// Test and measurement of the MetalFX upscaling of the presented image (runtime/nvn/src/nvn/metalfx_scale.h, the code metal.m runs).
// A reference image stands for the screen; it is reduced (area average, as a lower render resolution) and brought back to the
// screen's size by plain bilinear sampling (the present pass without the upscaler) and by MetalFX. Prints PSNR and SSIM of both
// against the reference and the GPU time of the scaler at the sizes the game shows on a Mac and an iPad.
// Checks: the scaler is set up only for an upscale, its output has the requested size and is close to the reference, and a
// repeated present of the same frame is not scaled again.
// Usage: metalfx_scale_test [IMAGE [OUT_DIR]]   IMAGE: a screenshot (PNG/JPEG; default: a synthetic test pattern); OUT_DIR: PNGs
// Needs an Apple silicon Mac (or any GPU with MetalFX).
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "metalfx_scale.h"

static int failures;
#define CHECK(c, ...) do { if (!(c)) { failures++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

typedef struct { int w, h; uint8_t *px; } Img;   // RGBA8, gamma-encoded

static Img img_new(int w, int h) { Img m = {w, h, calloc((size_t)w * h, 4)}; return m; }

static Img load_image(const char *path) {
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:@(path)], NULL);
    CGImageRef im = src ? CGImageSourceCreateImageAtIndex(src, 0, NULL) : NULL;
    if (!im) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    Img m = img_new((int)CGImageGetWidth(im), (int)CGImageGetHeight(im));
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef ctx = CGBitmapContextCreate(m.px, m.w, m.h, 8, (size_t)m.w * 4, cs, (CGBitmapInfo)kCGImageAlphaPremultipliedLast);
    CGContextDrawImage(ctx, CGRectMake(0, 0, m.w, m.h), im);
    CGContextRelease(ctx); CGColorSpaceRelease(cs); CGImageRelease(im); CFRelease(src);
    return m;
}

static void save_png(Img m, NSString *path) {
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef ctx = CGBitmapContextCreate(m.px, m.w, m.h, 8, (size_t)m.w * 4, cs, (CGBitmapInfo)kCGImageAlphaPremultipliedLast);
    CGImageRef im = CGBitmapContextCreateImage(ctx);
    CGImageDestinationRef d = CGImageDestinationCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:path], CFSTR("public.png"), 1, NULL);
    CGImageDestinationAddImage(d, im, NULL);
    CGImageDestinationFinalize(d);
    CFRelease(d); CGImageRelease(im); CGContextRelease(ctx); CGColorSpaceRelease(cs);
}

// Synthetic scene: soft gradients, hard diagonal edges, thin lines and small text-like strokes (what upscaling blurs first).
static Img pattern(int w, int h) {
    Img m = img_new(w, h);
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        uint8_t *p = m.px + ((size_t)y * w + x) * 4;
        double fx = (double)x / w, fy = (double)y / h;
        double r = 80 + 100 * fx, g = 120 + 80 * fy, b = 90 + 60 * sin(fx * 6.28);
        if ((int)((x + 2 * y) / 37) % 2) { r *= 0.6; g *= 0.7; }                       // diagonal bands
        double cx = x - w * 0.7, cy = y - h * 0.4;
        if (fabs(sqrt(cx * cx + cy * cy) - h * 0.2) < 2.0) r = g = b = 250;             // thin circle
        if (y > h * 0.75 && y < h * 0.85 && (x / 3) % 4 == 0 && ((x / 12 + y / 5) % 3)) r = g = b = 20;   // glyph-like strokes
        p[0] = (uint8_t)fmin(255, r); p[1] = (uint8_t)fmin(255, g); p[2] = (uint8_t)fmin(255, b); p[3] = 255;
    }
    return m;
}

// Area average to w x h (a lower render resolution of the same scene).
static Img reduce(Img s, int w, int h) {
    Img d = img_new(w, h);
    double sx = (double)s.w / w, sy = (double)s.h / h;
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        double acc[4] = {0}, wsum = 0;
        double x0 = x * sx, x1 = x0 + sx, y0 = y * sy, y1 = y0 + sy;
        for (int yy = (int)y0; yy < (int)ceil(y1) && yy < s.h; yy++) for (int xx = (int)x0; xx < (int)ceil(x1) && xx < s.w; xx++) {
            double wx = fmin(xx + 1, x1) - fmax(xx, x0), wy = fmin(yy + 1, y1) - fmax(yy, y0), wgt = wx * wy;
            const uint8_t *p = s.px + ((size_t)yy * s.w + xx) * 4;
            for (int c = 0; c < 4; c++) acc[c] += p[c] * wgt;
            wsum += wgt;
        }
        for (int c = 0; c < 4; c++) d.px[((size_t)y * w + x) * 4 + c] = (uint8_t)lround(acc[c] / wsum);
    }
    return d;
}

// Bilinear sampling at texel centres with clamped edges: what the present pass does without the upscaler.
static Img bilinear(Img s, int w, int h) {
    Img d = img_new(w, h);
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        double u = (x + 0.5) * s.w / w - 0.5, v = (y + 0.5) * s.h / h - 0.5;
        int x0 = (int)floor(u), y0 = (int)floor(v);
        double fx = u - x0, fy = v - y0;
        int xa = x0 < 0 ? 0 : x0, xb = x0 + 1 >= s.w ? s.w - 1 : x0 + 1, ya = y0 < 0 ? 0 : y0, yb = y0 + 1 >= s.h ? s.h - 1 : y0 + 1;
        if (xb < 0) xb = 0;
        if (yb < 0) yb = 0;
        for (int c = 0; c < 4; c++) {
            double a = s.px[((size_t)ya * s.w + xa) * 4 + c], b = s.px[((size_t)ya * s.w + xb) * 4 + c];
            double cc = s.px[((size_t)yb * s.w + xa) * 4 + c], dd = s.px[((size_t)yb * s.w + xb) * 4 + c];
            d.px[((size_t)y * w + x) * 4 + c] = (uint8_t)lround((a * (1 - fx) + b * fx) * (1 - fy) + (cc * (1 - fx) + dd * fx) * fy);
        }
    }
    return d;
}

static double psnr(Img a, Img b) {
    double se = 0;
    for (size_t i = 0; i < (size_t)a.w * a.h; i++) for (int c = 0; c < 3; c++) { double e = (double)a.px[i * 4 + c] - b.px[i * 4 + c]; se += e * e; }
    double mse = se / ((double)a.w * a.h * 3);
    return mse > 0 ? 10 * log10(255.0 * 255.0 / mse) : 99;
}

// Mean SSIM of the luma over 8x8 windows (stride 4).
static double ssim(Img a, Img b) {
    const double c1 = 6.5025, c2 = 58.5225;
    double sum = 0; int n = 0;
    for (int y = 0; y + 8 <= a.h; y += 4) for (int x = 0; x + 8 <= a.w; x += 4) {
        double ma = 0, mb = 0, va = 0, vb = 0, cov = 0;
        double la[64], lb[64];
        for (int j = 0; j < 8; j++) for (int i = 0; i < 8; i++) {
            const uint8_t *p = a.px + ((size_t)(y + j) * a.w + x + i) * 4, *q = b.px + ((size_t)(y + j) * b.w + x + i) * 4;
            la[j * 8 + i] = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
            lb[j * 8 + i] = 0.299 * q[0] + 0.587 * q[1] + 0.114 * q[2];
            ma += la[j * 8 + i]; mb += lb[j * 8 + i];
        }
        ma /= 64; mb /= 64;
        for (int k = 0; k < 64; k++) { va += (la[k] - ma) * (la[k] - ma); vb += (lb[k] - mb) * (lb[k] - mb); cov += (la[k] - ma) * (lb[k] - mb); }
        va /= 63; vb /= 63; cov /= 63;
        sum += ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
        n++;
    }
    return sum / n;
}

static id<MTLTexture> upload(id<MTLDevice> dev, Img m) {
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB width:m.w height:m.h mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;   // as a swapchain image
    id<MTLTexture> t = [dev newTextureWithDescriptor:td];
    [t replaceRegion:MTLRegionMake2D(0, 0, m.w, m.h) mipmapLevel:0 withBytes:m.px bytesPerRow:(NSUInteger)m.w * 4];
    return t;
}

static Img download(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLTexture> t) {
    Img m = img_new((int)t.width, (int)t.height);
    id<MTLBuffer> buf = [dev newBufferWithLength:(NSUInteger)m.w * m.h * 4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(t.width, t.height, 1)
               toBuffer:buf destinationOffset:0 destinationBytesPerRow:(NSUInteger)m.w * 4 destinationBytesPerImage:(NSUInteger)m.w * m.h * 4];
    [bl endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(m.px, buf.contents, (size_t)m.w * m.h * 4);
    return m;
}

// metal.m passes a raw (non-sRGB) view of the swapchain image
static id<MTLTexture> raw_view(id<MTLTexture> t) { return [t newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm]; }

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev || ![MTLFXSpatialScalerDescriptor supportsDevice:dev]) { printf("SKIP: no MetalFX spatial scaler on this GPU\n"); return 0; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        Img ref = argc > 1 ? load_image(argv[1]) : pattern(1280, 720);
        NSString *out = argc > 2 ? @(argv[2]) : nil;
        if (out) [NSFileManager.defaultManager createDirectoryAtPath:out withIntermediateDirectories:YES attributes:nil error:nil];
        printf("reference %dx%d (%s)\n", ref.w, ref.h, argc > 1 ? argv[1] : "synthetic pattern");

        // No upscale: no scaler
        id<MTLTexture> same = raw_view(upload(dev, ref));
        CHECK(metalfx_prepare(dev, same, ref.w, ref.h) == nil, "an image shown at its own size must not be upscaled");

        // Quality: render scale -> screen. 1.24: 1080p on an 11" iPad (1343 px tall at 16:9); 1.5: 720p in a 1080p window;
        // 1.87: 720p on that iPad; 2: 720p in a 1440p window.
        static const double ratios[] = {1.24, 1.5, 1.87, 2.0};
        printf("%-6s %-10s %18s %18s\n", "scale", "input", "bilinear PSNR/SSIM", "MetalFX PSNR/SSIM");
        for (int r = 0; r < 4; r++) {
            int iw = (int)lround(ref.w / ratios[r]), ih = (int)lround(ref.h / ratios[r]);
            Img low = reduce(ref, iw, ih);
            Img bl = bilinear(low, ref.w, ref.h);
            id<MTLTexture> in = raw_view(upload(dev, low));
            id<MTLTexture> o = metalfx_prepare(dev, in, ref.w, ref.h);
            CHECK(o && o.width == (NSUInteger)ref.w && o.height == (NSUInteger)ref.h, "scaler output %dx%d", ref.w, ref.h);
            if (!o) continue;
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLTexture> res = metalfx_encode(cb, in, 100 + r);
            [cb commit];
            [cb waitUntilCompleted];
            Img fx = download(dev, q, res);
            double pb = psnr(ref, bl), pf = psnr(ref, fx), sb = ssim(ref, bl), sf = ssim(ref, fx);
            printf("%-6.2f %4dx%-5d %9.2f dB %.4f %9.2f dB %.4f\n", ratios[r], iw, ih, pb, sb, pf, sf);
            CHECK(pf > 20 && sf > 0.6, "MetalFX output far from the reference (PSNR %.2f, SSIM %.4f)", pf, sf);
            if (out) {
                save_png(bl, [out stringByAppendingFormat:@"/x%.2f_bilinear.png", ratios[r]]);
                save_png(fx, [out stringByAppendingFormat:@"/x%.2f_metalfx.png", ratios[r]]);
            }
            free(low.px); free(bl.px); free(fx.px);
        }

        // A repeated present of the same frame (new view of the same image) encodes nothing; the next frame does.
        {
            Img low = reduce(ref, ref.w * 2 / 3, ref.h * 2 / 3);
            id<MTLTexture> img = upload(dev, low);
            id<MTLTexture> in = raw_view(img);
            metalfx_prepare(dev, in, ref.w, ref.h);
            id<MTLCommandBuffer> cb = [q commandBuffer];
            metalfx_encode(cb, in, 7);
            [cb commit]; [cb waitUntilCompleted];
            id<MTLTexture> in2 = raw_view(img);
            metalfx_prepare(dev, in2, ref.w, ref.h);
            g_fx.colorTexture = nil;
            id<MTLCommandBuffer> cb2 = [q commandBuffer];
            metalfx_encode(cb2, in2, 7);
            CHECK(g_fx.colorTexture == nil, "a repeated present of frame 7 was scaled again");
            [cb2 commit]; [cb2 waitUntilCompleted];
            id<MTLCommandBuffer> cb3 = [q commandBuffer];
            metalfx_encode(cb3, in2, 8);
            CHECK(g_fx.colorTexture == in2, "frame 8 was not scaled");
            [cb3 commit]; [cb3 waitUntilCompleted];
            free(low.px);
        }

        // GPU time of the scaler (median of 40 runs, each its own command buffer) at the sizes the game shows
        static const int sizes[][4] = {
            {1280, 720, 2388, 1343},   // 720p on an 11" iPad Pro / Air (2388x1668 panel, 16:9 picture)
            {1920, 1080, 2388, 1343},  // 1080p on the same iPad
            {1280, 720, 2732, 1537},   // 720p on a 12.9" iPad Pro (2732x2048)
            {1920, 1080, 2732, 1537},
            {1280, 720, 2560, 1440},   // 720p in a 1440p Mac window
            {1920, 1080, 2560, 1440},
            {1920, 1080, 3456, 1944},  // 1080p on a 16" MacBook Pro panel
        };
        printf("\nMetalFX GPU time on %s:\n", dev.name.UTF8String);
        for (unsigned s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            Img low = reduce(ref, sizes[s][0], sizes[s][1]);
            id<MTLTexture> in = raw_view(upload(dev, low));
            double t[40];
            for (int i = 0; i < 40; i++) {
                metalfx_prepare(dev, in, sizes[s][2], sizes[s][3]);
                id<MTLCommandBuffer> cb = [q commandBuffer];
                metalfx_encode(cb, in, 1000 + s * 100 + i);
                [cb commit];
                [cb waitUntilCompleted];
                t[i] = (cb.GPUEndTime - cb.GPUStartTime) * 1000;
            }
            for (int i = 0; i < 40; i++) for (int j = i + 1; j < 40; j++) if (t[j] < t[i]) { double x = t[i]; t[i] = t[j]; t[j] = x; }
            double mp_in = sizes[s][0] * sizes[s][1] / 1e6, mp_out = sizes[s][2] * sizes[s][3] / 1e6;
            printf("  %4dx%-4d -> %4dx%-4d  %.3f ms  (render %.2f MP instead of %.2f MP: %.0f%% fewer pixels)\n", sizes[s][0], sizes[s][1],
                   sizes[s][2], sizes[s][3], t[20], mp_in, mp_out, 100 * (1 - mp_in / mp_out));
            free(low.px);
        }
        free(ref.px);
    }
    printf(failures ? "\n%d FAILED\n" : "\nOK\n", failures);
    return failures ? 1 : 0;
}

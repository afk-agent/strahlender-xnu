// MetalFX temporal upscaler, the renderer's part (setting "upscaler = temporal"; the scaler itself runs at the present: metalfx_temporal.h).
// Included by renderer.m, so it shares its state (st, enc, dev, cb(), end_pass, LOG, ...).
//
// A temporal upscaler needs three things the game does not make on its own:
//  - jitter: every frame the 3D scene is drawn shifted by a different sub-pixel offset (a Halton sequence), so that frames together
//    sample each output pixel at many places. A draw is jittered when it tests depth against a depth target of the presented
//    image's size (the depth pre-pass, opaque and transparent scene geometry); full-screen passes (compare function "always") and
//    targets of other sizes (shadow maps, bloom) are not.
//  - motion: where each pixel was in the previous frame. Jittered draws run a variant of their vertex shader that also computes the
//    position with the previous frame's constant buffers (camera and object matrices: Unity keeps them at fixed addresses, so the
//    previous frame's contents of an address are its previous values). An opaque one in a pass whose colour target has the presented
//    size also writes previous - current (uv units) into an extra colour attachment. The previous contents are copies taken when a
//    jittered draw used them, per guest address and per version within the frame (an address rewritten between draws has several
//    versions, matched in order), so a camera buffer every draw shares is copied once per frame.
//    Vertex positions the GPU skins anew every frame (characters): a compute range that jittered draws read as a vertex stream is
//    learned, and before the next frame's dispatch overwrites it, it is copied (the previous frame's skinned vertices); the second
//    run of the vertex shader reads its attributes from that copy (ShaderKey.motion_prev_attrs). Not with the arena (devices below
//    5 GB, where compute writes go through copies): there they move only with their object matrix.
//  - pixels without usable history: the HUD, text and menus are drawn over the scene's composite into the presented image. The image
//    is copied before the first such draw; at the present, pixels that differ from the copy are "reactive" (MetalFX follows the
//    current frame there).
// Depth and motion come from the scene's own target, which the game draws upside down relative to the presented image (Unity's
// render-texture convention); the present turns them (metalfx_temporal.h).

typedef struct { uint64_t addr; uint32_t ver, size, off, gen; } TaaEnt;   // one copied version of a constant buffer
#define TAA_TAB 8192
#define TAA_ARENA_MAX (8u << 20)
typedef struct {
    TaaEnt e[TAA_TAB];                                            // (addr, ver) -> copy
    struct { uint64_t addr; uint32_t nver, off, size, gen; } last[TAA_TAB];   // addr -> versions this frame and the newest one
    uint8_t *data;
    uint32_t len, cap, gen, n;
} TaaArena;
static TaaArena *taa_cur, *taa_prev;

static struct {
    bool on;                 // this frame records motion (temporal upscaling or frame generation)
    bool jitter;             // ... and is jittered (temporal upscaling)
    bool broken;             // the present could not run the scaler: no jitter until the setting changes
    uint64_t frame;          // frame number of the state below (counts presents)
    uint32_t w, h;           // guest size of the presented image; targets of that size get jitter and motion
    uint32_t phase, nphase;  // jitter sequence
    float dx, dy;            // jitter of this frame in pixels (shift of the content, y down in the target)
    float out_scale;         // output / input of the last scaled frame (more phases for larger scales)
    id<MTLTexture> mv;       // motion (RG16Float, previous - current in uv units), the presented image's size
    id<MTLTexture> mv_ms;    // the attachment of multisampled passes (cutscenes draw with 4x MSAA), resolved into mv
    uint64_t mv_frame, mv_ms_frame;   // frame the motion texture / its multisampled attachment was last cleared in
    bool mv_written;
    id<MTLTexture> depth;    // the scene's depth: the depth target of the pass the motion was written in
    id<MTLTexture> pass_depth, pass_color;
    id<MTLTexture> scene;    // the colour target of the passes with motion (metal.m checks the scene's orientation against the image)
    float depth_clear;       // value the game clears it to (0: reversed depth)
    NvnTexture *wants[8];    // colour targets that had opaque jittered draws (their passes get the motion attachment)
    uint64_t wants_frame[8];
    NvnTexture *composite;   // presented image the scene was composited into this frame
    id<MTLTexture> copy;     // that image before the first draw over the composite
    id<MTLTexture> copy_src;
    id<MTLBuffer> prev_buf[4];   // the previous frame's constant buffers on the GPU (one per frame in flight)
    __strong id<MTLCommandBuffer> prev_cb[4];
    int prev_i;
    uint64_t prev_frame;
    uint64_t draws, mv_draws, hits, misses, skin_copies, skin_bytes, skin_draws;   // (stats)
    uint32_t off_frames;     // frames since the effects went off (their textures and copies are released after a second)
    struct { uint64_t addr, size, seen, snap; id<MTLBuffer> prev; } skin[32];   // compute ranges read as vertex streams; their copies
    int draw_skin;           // (taa_keys -> draw) skin entry of the draw's previous-frame attributes, -1: none
    int draw_skin_stream;    // the vertex stream they replace
    uint64_t draw_skin_off;  // offset of the draw's stream in that copy
} taa;
// (pass_mv_slot, renderer.m: colour attachment of the motion in the open pass, -1: none)
static bool pass_taa;           // the open pass's depth target has the presented image's size

static double taa_halton(uint32_t i, uint32_t b) { double f = 1, r = 0; for (i++; i > 0; i /= b) { f /= b; r += f * (i % b); } return r; }

static TaaArena *taa_arena_new(void) { TaaArena *a = calloc(1, sizeof *a); a->gen = 1; return a; }
static void taa_arena_reset(TaaArena *a) {
    a->len = a->n = 0;
    if (++a->gen == 0) { memset(a->e, 0, sizeof a->e); memset(a->last, 0, sizeof a->last); a->gen = 1; }
}
static uint32_t taa_hash(uint64_t a, uint32_t v) {
    uint64_t h = ((a >> 4) + (uint64_t)v * 0x632BE59BD9B4E019ull) * 0x9E3779B97F4A7C15ull;
    return (uint32_t)(h >> 40) & (TAA_TAB - 1);
}
static TaaEnt *taa_find(TaaArena *a, uint64_t addr, uint32_t ver) {
    for (uint32_t i = taa_hash(addr, ver), n = 0; n < TAA_TAB; i = (i + 1) & (TAA_TAB - 1), n++) {
        TaaEnt *e = &a->e[i];
        if (e->gen != a->gen) return NULL;
        if (e->addr == addr && e->ver == ver) return e;
    }
    return NULL;
}

// Copy of the constant buffer at addr (size bytes) for this frame: a new version when its contents changed since the last copy of
// the address in this frame. Returns the version (-1: the table or the arena is full).
static int taa_snapshot(uint64_t addr, uint32_t size) {
    TaaArena *a = taa_cur;
    uint32_t li = taa_hash(addr, 0xFFFFFFFFu);
    for (uint32_t n = 0; n < TAA_TAB && a->last[li].gen == a->gen && a->last[li].addr != addr; n++) li = (li + 1) & (TAA_TAB - 1);
    bool seen = a->last[li].gen == a->gen && a->last[li].addr == addr;
    if (seen && a->last[li].size == size && !memcmp(a->data + a->last[li].off, (const void *)(uintptr_t)addr, size))
        return (int)a->last[li].nver - 1;
    if (a->n >= TAA_TAB * 3 / 4 || a->len + size + 255 > TAA_ARENA_MAX || (!seen && a->last[li].gen == a->gen)) return -1;
    if (a->len + size + 255 > a->cap) {
        uint32_t cap = a->cap ? a->cap : 256 << 10;
        while (cap < a->len + size + 255) cap *= 2;
        a->data = realloc(a->data, cap);
        a->cap = cap;
    }
    uint32_t off = (a->len + 255) & ~255u;   // (constant buffer offsets: 256-byte aligned like Metal's on macOS)
    memcpy(a->data + off, (const void *)(uintptr_t)addr, size);
    a->len = off + size;
    uint32_t ver = seen ? a->last[li].nver : 0;
    a->last[li].addr = addr; a->last[li].nver = ver + 1; a->last[li].off = off; a->last[li].size = size; a->last[li].gen = a->gen;
    uint32_t i = taa_hash(addr, ver);
    while (a->e[i].gen == a->gen) i = (i + 1) & (TAA_TAB - 1);
    a->e[i] = (TaaEnt){addr, ver, size, off, a->gen};
    a->n++;
    return (int)ver;
}

// Frame boundary (the present of a frame, in the worker's order): this frame's copies become the previous ones.
static void taa_next_frame(void) {
    if (!taa_cur) { taa_cur = taa_arena_new(); taa_prev = taa_arena_new(); }
    TaaArena *t = taa_prev; taa_prev = taa_cur; taa_cur = t;
    taa_arena_reset(taa_cur);
    taa.frame++;
    taa.composite = NULL;
    taa.copy_src = nil;
    taa.mv_written = false;
    taa.depth = nil;
    taa.scene = nil;
    bool want = __atomic_load_n(&g_settings.upscaler, __ATOMIC_RELAXED) == UPSCALE_METALFX_TEMPORAL;
    extern volatile int g_bdsp_framegen_active;   // (metal.m: frame generation needs the motion and depth too, not the jitter)
    if (!want) taa.broken = false;   // (switching the setting off and on tries again)
    taa.jitter = want && !taa.broken && taa.w && taa.h;
    taa.on = (taa.jitter || g_bdsp_framegen_active) && taa.w && taa.h;
    if (taa.on) taa.off_frames = 0;
    else if (++taa.off_frames == 60 && (taa.mv || taa.copy || taa_cur->cap)) {   // (the motion textures, the pre-HUD copy, the arenas)
        taa.mv = taa.mv_ms = taa.copy = nil;
        for (int i = 0; i < 4; i++) { taa.prev_buf[i] = nil; taa.prev_cb[i] = nil; }
        for (TaaArena *a = taa_cur; a; a = a == taa_cur ? taa_prev : NULL) { free(a->data); a->data = NULL; a->cap = 0; taa_arena_reset(a); }
        memset(taa.wants, 0, sizeof taa.wants);
        LOG("[taa] off: motion textures and buffer copies released\n");
    }
    // MetalFX's guidance: at least 8 phases per input pixel, times the square of the scale
    uint32_t n = (uint32_t)ceilf(8.0f * fmaxf(1.0f, taa.out_scale * taa.out_scale));
    taa.nphase = n > 64 ? 64 : n;
    taa.phase = (taa.phase + 1) % taa.nphase;
    taa.dx = taa.jitter ? (float)(taa_halton(taa.phase, 2) - 0.5) : 0;
    taa.dy = taa.jitter ? (float)(taa_halton(taa.phase, 3) - 0.5) : 0;
}

// The depth target of the current state has the presented image's size (jitter and motion apply)
static bool taa_depth_target(void) {
    if (!taa.on || !st.rt.depth) return false;
    NvnTexture *d = st.rt.depth;
    return d->width == taa.w && d->height == taa.h && renderer_format_is_depth(d->format) && d->format != 0x31;
}

// begin_pass: a pass whose colour target had opaque jittered draws in the last frames gets the motion texture as an extra attachment
static void taa_begin_pass(MTLRenderPassDescriptor *rp) {
    pass_mv_slot = -1;
    pass_taa = taa_depth_target();
    taa.pass_depth = taa.pass_color = nil;
    if (!pass_taa) return;
    taa.pass_depth = rp.depthAttachment.texture;
    taa.pass_color = rp.colorAttachments[0].texture;
    NvnTexture *c = st.rt.ncolor ? st.rt.color[0] : NULL;
    if (!c || st.rt.ncolor >= 8 || c->width != taa.w || c->height != taa.h || c->samples != st.rt.depth->samples ||
        renderer_texture_presented(c)) return;
    bool want = false;
    for (int i = 0; i < 8; i++) want |= taa.wants[i] == c && taa.wants_frame[i] + 2 >= taa.frame;
    if (!want) return;
    if (!taa.mv || taa.mv.width != taa.w || taa.mv.height != taa.h) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float width:taa.w
                                                                                    height:taa.h mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        taa.mv = [dev newTextureWithDescriptor:td];
        taa.mv.label = @"taa motion";
    }
    int slot = st.rt.ncolor;
    uint64_t *cleared = c->samples > 1 ? &taa.mv_ms_frame : &taa.mv_frame;
    rp.colorAttachments[slot].loadAction = *cleared == taa.frame ? MTLLoadActionLoad : MTLLoadActionClear;
    rp.colorAttachments[slot].clearColor = MTLClearColorMake(0, 0, 0, 0);
    *cleared = taa.frame;
    if (c->samples > 1) {   // multisampled: drawn per sample, resolved (averaged) into the motion texture
        if (!taa.mv_ms || taa.mv_ms.width != taa.w || taa.mv_ms.height != taa.h || taa.mv_ms.sampleCount != c->samples) {
            MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float width:taa.w
                                                                                        height:taa.h mipmapped:NO];
            td.textureType = MTLTextureType2DMultisample;
            td.sampleCount = c->samples;
            td.usage = MTLTextureUsageRenderTarget;
            td.storageMode = MTLStorageModePrivate;
            taa.mv_ms = [dev newTextureWithDescriptor:td];
            taa.mv_ms.label = @"taa motion (multisampled)";
        }
        // (a later multisampled pass of the frame loads the samples; a single-sampled one the resolved texture)
        rp.colorAttachments[slot].texture = taa.mv_ms;
        rp.colorAttachments[slot].resolveTexture = taa.mv;
        rp.colorAttachments[slot].storeAction = MTLStoreActionStoreAndMultisampleResolve;
    } else {
        rp.colorAttachments[slot].texture = taa.mv;
        rp.colorAttachments[slot].storeAction = MTLStoreActionStore;
    }
    pass_mv_slot = slot;
}

// Every pipeline used in a pass with the motion attachment lists it; only draws with the motion output write it.
static void taa_attachment_format(MTLRenderPipelineDescriptor *pd) {
    if (pass_mv_slot < 0) return;
    pd.colorAttachments[pass_mv_slot].pixelFormat = MTLPixelFormatRG16Float;
    pd.colorAttachments[pass_mv_slot].writeMask = MTLColorWriteMaskNone;
}

static void taa_note_clear(bool depth, float dval) {
    if (depth && pass_taa) taa.depth_clear = dval;
}

// Shader variant of a draw (chosen before begin_pass): jitter and the previous position when it tests depth against a target of the
// presented size; the motion output when it does not blend and its pass will have the motion attachment.
static uint16_t taa_skin_attrs(const ShaderKey *vkey);
static void taa_keys(ShaderKey *vkey, ShaderKey *fkey) {
    if (!taa_depth_target() || !st.ds[0] || st.ds[2] == 8 /* always */ || vkey->flip_x) return;
    // (not into the presented image: what is drawn there after the composite is reactive, so MetalFX shows it as drawn - jittered)
    if (st.rt.ncolor && st.rt.color[0] && renderer_texture_presented(st.rt.color[0])) return;
    vkey->motion = 1;
    vkey->motion_prev_attrs = taa_skin_attrs(vkey);   // (GPU-skinned vertices: the previous frame's copy)
    static int bias = -1;   // BDSP_TAA_LOD_BIAS=0: textures keep the level of detail of the render resolution
    if (bias < 0) bias = !getenv("BDSP_TAA_LOD_BIAS") || atoi(getenv("BDSP_TAA_LOD_BIAS")) != 0;
    fkey->lod_bias = bias && taa.jitter && taa.out_scale > 1.01f;
    if (st.color_state[0] & 1) return;
    int slot = pass_mv_slot;
    if (!enc || st.rt_dirty) {   // the pass opens with this draw: it gets the attachment as taa_begin_pass decides
        NvnTexture *c = st.rt.ncolor ? st.rt.color[0] : NULL;
        bool want = false;
        if (c && st.rt.ncolor < 8 && c->width == taa.w && c->height == taa.h && c->samples == st.rt.depth->samples && !renderer_texture_presented(c))
            for (int i = 0; i < 8; i++) want |= taa.wants[i] == c && taa.wants_frame[i] + 2 >= taa.frame;
        slot = want ? st.rt.ncolor : -1;
    }
    if (slot >= 0) fkey->motion = (uint8_t)(1 + slot);
}

// Texture level-of-detail bias of the jittered draws' fragment shaders: log2(render / output) (-1 for a 2x upscale), as the scene
// is sampled at the output's density over the frames; the usual setting for temporal upscalers.
static void taa_bind_fragment(Prog *pr) {
    float b[4] = {-log2f(fmaxf(1.0f, taa.out_scale)), 0, 0, 0};
    if (b[0] < -2) b[0] = -2;
    [enc setFragmentBytes:b length:sizeof b atIndex:MV_PARAMS_BUFFER_INDEX];
}

// After begin_pass, for a draw with the jitter variant: an opaque one marks its colour target for the motion attachment (from the
// target's next pass on); one that writes motion makes its pass's depth the scene's depth.
static void taa_note_draw(Prog *pr) {
    taa.draws++;
    NvnTexture *c = st.rt.ncolor ? st.rt.color[0] : NULL;
    if (c && !(st.color_state[0] & 1) && c->width == taa.w && c->height == taa.h && !renderer_texture_presented(c)) {
        int i, free_i = -1;
        for (i = 0; i < 8 && taa.wants[i] != c; i++)
            if (free_i < 0 && (!taa.wants[i] || taa.wants_frame[i] + 8 < taa.frame)) free_i = i;
        if (i == 8 && free_i >= 0) { i = free_i; taa.wants[i] = c; }
        if (i < 8) taa.wants_frame[i] = taa.frame;
    }
    if (pr->fi->motion_written && pass_mv_slot >= 0) {
        taa.mv_written = true;
        taa.mv_draws++;
        taa.depth = taa.pass_depth;
        taa.scene = taa.pass_color;
    }
}

// ---- GPU-skinned vertices: the previous frame's copy of a compute range read as a vertex stream
static bool taa_skin_possible(void) {   // (BDSP_TAA_SKIN=0: no copies, skinned characters move with their object matrix only)
    static int on = -1;
    if (on < 0) on = !getenv("BDSP_TAA_SKIN") || atoi(getenv("BDSP_TAA_SKIN")) != 0;
    return on && taa.on && !arena_on();
}

// (draw) a jittered draw reads stream [a, a + size): a compute range it lies in is remembered for copying
static void taa_skin_learn(uint64_t a, uint64_t size) {
    for (int k = 0; k < hz_n; k++) {
        if (a < hz_ranges[k].addr || a + size > hz_ranges[k].addr + hz_ranges[k].size) continue;
        int i, free_i = -1;
        for (i = 0; i < 32; i++) {
            if (taa.skin[i].addr == hz_ranges[k].addr && taa.skin[i].size == hz_ranges[k].size) break;
            if (free_i < 0 && (!taa.skin[i].addr || taa.skin[i].seen + 60 < taa.frame)) free_i = i;
        }
        if (i == 32) {
            if (free_i < 0) return;
            i = free_i;
            taa.skin[i].addr = hz_ranges[k].addr; taa.skin[i].size = hz_ranges[k].size; taa.skin[i].snap = 0; taa.skin[i].prev = nil;
        }
        taa.skin[i].seen = taa.frame;
        return;
    }
}

// (dispatch_compute, before the dispatch is encoded) a learned range this dispatch writes is copied first: the copy holds what the
// previous frame's dispatch wrote. (A range written twice in a frame gets the first write as its "previous" one: no own motion.)
static void taa_skin_before_dispatch(void) {
    if (!taa_skin_possible()) return;
    id<MTLBlitCommandEncoder> bl = nil;
    for (int r = 0; r < 16; r++) {
        uint64_t a = st.sbuf[5][r].addr, size = st.sbuf[5][r].size;
        if (!a || !size || a < HEAP_BASE || a + size > HEAP_BASE + HEAP_MAX || (a & 3) || (size & 3)) continue;
        for (int i = 0; i < 32; i++) {
            if (taa.skin[i].addr != a || taa.skin[i].size != size || taa.skin[i].seen + 4 < taa.frame || taa.skin[i].snap == taa.frame) continue;
            if (!taa.skin[i].prev || taa.skin[i].prev.length < size)
                taa.skin[i].prev = [dev newBufferWithLength:size options:MTLResourceStorageModePrivate];
            if (!bl) bl = [cb() blitCommandEncoder];
            [bl copyFromBuffer:gpu_guest_buffer() sourceOffset:a - HEAP_BASE toBuffer:taa.skin[i].prev destinationOffset:0 size:size];
            taa.skin[i].snap = taa.frame;
            taa.skin_copies++; taa.skin_bytes += size;
        }
    }
    if (bl) [bl endEncoding];
}

// (taa_keys) the attributes of the draw that come from a stream with a copy made this frame: they read the copy in the second run
static uint16_t taa_skin_attrs(const ShaderKey *vkey) {
    taa.draw_skin = -1;
    if (!taa_skin_possible()) return 0;
    for (int i = 0; i < st.nattrib && i < 16; i++) {
        uint32_t a = st.attrib[i], s = (a >> 8) & 0x1F;
        if (!(a & 0xFF) || s >= 12 || !st.vbuf[s].addr || s >= (uint32_t)st.nstream || !st.stream[s][0] || st.stream[s][1]) continue;
        uint64_t va = st.vbuf[s].addr, vs = st.vbuf[s].size ? st.vbuf[s].size : 16;
        taa_skin_learn(va, vs);
        for (int k = 0; k < 32 && taa.draw_skin < 0; k++)
            if (taa.skin[k].snap == taa.frame && taa.skin[k].prev && va >= taa.skin[k].addr && va + vs <= taa.skin[k].addr + taa.skin[k].size &&
                !((va - taa.skin[k].addr) & 3)) {
                taa.draw_skin = k;
                taa.draw_skin_off = va - taa.skin[k].addr;
                taa.draw_skin_stream = (int)s;
            }
    }
    if (taa.draw_skin < 0) return 0;
    taa.skin_draws++;
    uint16_t m = 0;
    for (int i = 0; i < st.nattrib && i < 16; i++)
        if ((st.attrib[i] & 0xFF) && (int)((st.attrib[i] >> 8) & 0x1F) == taa.draw_skin_stream && vkey->attr_kind[i] == 0) m |= 1u << i;
    return m;
}

// The previous frame's copies on the GPU, uploaded once per frame (a pool of buffers, one per frame in flight)
static id<MTLBuffer> taa_prev_buffer(void) {
    int i = taa.prev_i;
    if (taa.prev_frame != taa.frame || !taa.prev_buf[i]) {
        i = taa.prev_i = (taa.prev_i + 1) % 4;
        id<MTLCommandBuffer> last = taa.prev_cb[i];   // (four frames ago: committed by its present long since)
        if (last && last.status >= MTLCommandBufferStatusCommitted && last.status < MTLCommandBufferStatusCompleted) [last waitUntilCompleted];
        else if (last && last.status < MTLCommandBufferStatusCommitted) taa.prev_buf[i] = nil;   // (never wait for an open one: new buffer)
        uint32_t need = taa_prev->len > 256 ? taa_prev->len : 256;
        if (!taa.prev_buf[i] || taa.prev_buf[i].length < need)
            taa.prev_buf[i] = [dev newBufferWithLength:(need + 0xFFFF) & ~0xFFFFu options:MTLResourceStorageModeShared];
        if (taa_prev->len) memcpy(taa.prev_buf[i].contents, taa_prev->data, taa_prev->len);
        taa.prev_frame = taa.frame;
    }
    taa.prev_cb[i] = cb();
    return taa.prev_buf[i];
}

// Binds the motion variant's parameters: the jitter and, per constant buffer of the vertex stage, where the previous frame's version
// is (~0: this frame's, for a buffer the previous frame did not use).
static void taa_bind(Prog *pr, int cbuf_base_, float vpw, float vph) {
    MvParams p;
    memset(&p, 0xFF, sizeof p);
    p.jitter[0] = vpw > 0 ? 2.0f * taa.dx / vpw : 0;
    p.jitter[1] = vph > 0 ? -2.0f * taa.dy / vph : 0;
    p.jitter[2] = p.jitter[3] = 0;
    ShaderInfo *si = pr->vi;
    for (int k = 3; k < 16; k++) {
        if (!(si->cbuf_mask & (1u << k))) continue;
        int idx = k - cbuf_base_;
        Range r = (idx >= 0 && idx < 18) ? st.ubuf[0][idx] : (Range){0, 0};
        if (!r.addr) continue;
        uint64_t size = si->cbuf_size[k];
        if ((si->cbuf_dynamic & (1u << k)) && r.size) size = r.size > 0x10000 ? 0x10000 : r.size;   // (as bind_cbufs)
        if (size > r.size && r.size) size = r.size;
        if (size < 16) size = 16;
        int ver = taa_snapshot(r.addr, (uint32_t)size);
        TaaEnt *e = ver >= 0 ? taa_find(taa_prev, r.addr, (uint32_t)ver) : NULL;
        if (e && e->size == size) { p.off[k] = e->off; taa.hits++; }
        else taa.misses++;
    }
    [enc setVertexBytes:&p length:sizeof p atIndex:MV_PARAMS_BUFFER_INDEX];
    [enc setVertexBuffer:taa_prev_buffer() offset:0 atIndex:MV_PREV_BUFFER_INDEX];
}

// Before a draw into the presented image (and before its pass opens): the composite of the scene marks the image; the first other
// draw over it takes the copy that the reactive mask compares with.
static void taa_presented_draw(CmdRec *r, Prog *pr) {
    NvnTexture *c = st.rt.ncolor ? st.rt.color[0] : NULL;
    if (!taa.on || !c || !renderer_texture_presented(c)) return;
    bool full = r->x[2] <= 6 && st.viewport[0][0] == 0 && st.viewport[0][1] == 0 && st.viewport[0][2] >= c->width &&
                st.viewport[0][3] >= c->height;
    bool composite = false;
    if (full && !(st.color_state[0] & 1) && pr->fi->ntex) {
        NvnTexture *t0 = shader_texture_object(1, &pr->fi->tex[0]);
        composite = t0 && t0->width == c->width && t0->height == c->height;
    }
    if (composite) { taa.composite = c; taa.copy_src = nil; return; }
    if (taa.composite != c || taa.copy_src) return;
    end_pass();   // (the next draw opens a new pass that loads the image)
    id<MTLTexture> src = view_texture(c, NULL, false, true, nil);
    if (!taa.copy || taa.copy.width != src.width || taa.copy.height != src.height || taa.copy.pixelFormat != src.pixelFormat) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:src.pixelFormat width:src.width
                                                                                    height:src.height mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
        td.storageMode = MTLStorageModePrivate;
        taa.copy = [dev newTextureWithDescriptor:td];
        taa.copy.label = @"taa scene copy";
    }
    id<MTLBlitCommandEncoder> bl = [cb() blitCommandEncoder];
    [bl copyFromTexture:src sourceSlice:0 sourceLevel:0 toTexture:taa.copy destinationSlice:0 destinationLevel:0 sliceCount:1 levelCount:1];
    [bl endEncoding];
    taa.copy_src = src;
}

// The present (metal.m, in the worker's order after the frame's work): what the scaler needs for the presented image `tex`, then the
// next frame. size: guest size of the presented image, which the next frame's targets are compared with.
void renderer_taa_frame(id<MTLTexture> tex, uint32_t gw, uint32_t gh, TaaFrame *out) {
    memset(out, 0, sizeof *out);
    if (taa.on) {
        out->active = true;
        out->motion = taa.mv_written ? taa.mv : nil;
        out->depth = taa.mv_written ? taa.depth : nil;
        out->scene = taa.mv_written ? taa.scene : nil;
        id<MTLTexture> base = tex.parentTexture ? tex.parentTexture : tex;
        id<MTLTexture> cs = taa.copy_src.parentTexture ? taa.copy_src.parentTexture : taa.copy_src;
        out->copy = taa.copy_src && cs == base ? taa.copy : nil;
        out->jitter_x = taa.dx;
        out->jitter_y = taa.dy;
        out->depth_reversed = taa.depth_clear == 0.0f;
        static int nlog;
        if (getenv("BDSP_TAA_LOG") && nlog++ % 120 == 0)
            LOG("[taa] frame %llu %ux%u jitter %.3f %.3f (%u/%u) | since start: jittered draws %llu, with motion output %llu, constant buffers "
                "with a previous version %llu, without %llu, skinned draws %llu (copies %llu, %llu KB) | this frame: motion %d depth %d "
                "pre-HUD copy %d, copies %u KB\n",
                (unsigned long long)taa.frame, taa.w, taa.h, taa.dx, taa.dy, taa.phase, taa.nphase, (unsigned long long)taa.draws,
                (unsigned long long)taa.mv_draws, (unsigned long long)taa.hits, (unsigned long long)taa.misses, (unsigned long long)taa.skin_draws,
                (unsigned long long)taa.skin_copies, (unsigned long long)(taa.skin_bytes >> 10), out->motion != nil,
                out->depth != nil, out->copy != nil, taa_cur ? taa_cur->len >> 10 : 0);
    }
    taa.w = gw; taa.h = gh;
    taa_next_frame();
}

// The present could not run the temporal scaler (or ran it at another scale): stop the jitter / adapt the jitter sequence.
void renderer_taa_status(bool ok, float out_scale) {
    if (!ok) taa.broken = true;
    taa.out_scale = out_scale;
}

// Replacement for runtime/settings.h: the upscaler (off, MetalFX spatial or temporal, BDSP_UPSCALER: nvn_glue.c settings_load); no SMAA, no
// fullscreen setting.
#pragma once
#include <stdbool.h>

enum { UPSCALE_OFF, UPSCALE_METALFX, UPSCALE_METALFX_TEMPORAL };
enum { AA_OFF, AA_SMAA };
typedef struct {
    int upscaler, antialias, res_mode;
    bool fullscreen;
    int upscale_max;     // largest output / render scale of the temporal upscaler in percent (0: the screen's size; BDSP_UPSCALE_MAX)
    int framegen;        // frame generation: the game runs at half the frame rate, MetalFX makes every other picture (BDSP_FRAMEGEN)
} BdspSettings;
extern BdspSettings g_settings;
void settings_load(void);

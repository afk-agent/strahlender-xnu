// Replacement for runtime/settings.h: the upscaler (off, MetalFX spatial or temporal, BDSP_UPSCALER: nvn_glue.c settings_load); no SMAA, no
// fullscreen setting.
#pragma once
#include <stdbool.h>

enum { UPSCALE_OFF, UPSCALE_METALFX, UPSCALE_METALFX_TEMPORAL };
enum { AA_OFF, AA_SMAA };
typedef struct { int upscaler, antialias, res_mode; bool fullscreen; } BdspSettings;
extern BdspSettings g_settings;
void settings_load(void);

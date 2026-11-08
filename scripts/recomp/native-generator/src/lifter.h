/* SPDX-License-Identifier: MIT */
#ifndef BDSP_LIFTER_H
#define BDSP_LIFTER_H
#include "codegen.h"
typedef enum { CG_OP, CG_TRAP, CG_B, CG_BCOND, CG_BL, CG_BLR, CG_BR, CG_RET, CG_ASM, CG_PENDING } InsnKind;
/* read/write: whether the instruction reads the guest register and whether it writes it (a written register is always written in
   full). The batch loads only registers read before being written and stores only written ones; unknown uses stay read + write. */
typedef struct { unsigned position, reg; bool read, write; } AsmRegister;
typedef struct { unsigned position; uint64_t value; bool negative; } AsmConstant;
typedef struct {
    uint32_t word, vregs;
    /* exact: vread/vwrite are the vector registers the instruction reads and writes, vfull those it writes in all 128 bits without
       reading them first (a partial write lists the register in vread too). Otherwise every register in vregs counts as read and
       written, which is always correct. */
    uint32_t vread, vwrite, vfull;
    bool exact;
    AsmRegister gprs[3]; size_t gpr_count;
    AsmConstant constants[1]; size_t constant_count;
    bool flags;
} AsmInstruction;
typedef struct {
    InsnKind kind;
    const char *code, *condition, *target_expression;
    uint64_t target;
    bool numeric_target, negative_target;
    AsmInstruction assembly;
} Instruction;
Instruction cg_lift_basic(uint32_t word, uint64_t pc, bool portable, bool has_previous, uint32_t previous);
bool cg_emit_asm_batch(const AsmInstruction *batch, size_t count, const char **result);
int cg_decode(int argc, char **argv);
#endif

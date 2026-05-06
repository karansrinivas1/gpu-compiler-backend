#include "simt/PTXEmitter.h"
#include <cassert>
#include <sstream>

namespace simt {
namespace codegen {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string ptxRegPrefix(ir::Type* ty) {
  if (ty->isPred())  return "%p";
  if (ty->isFloat()) return "%f";
  if (ty->bitWidth() == 64) return "%rd";
  return "%r";
}

std::string PTXEmitter::physRegName(const ir::Function& fn,
                                     ir::VRegID vid) const {
  const auto* vr = fn.vregs[vid].get();
  if (vr->isSpilled())
    return "[spill+" + std::to_string(vr->spillSlot) + "]";
  return ptxRegPrefix(vr->type) + std::to_string(vr->physReg);
}

void PTXEmitter::line(const std::string& s) {
  for (unsigned i = 0; i < indent_; ++i) out_ << "\t";
  out_ << s << "\n";
}
void PTXEmitter::line() { out_ << "\n"; }

std::string PTXEmitter::opcodeStr(ir::Opcode op) const {
  using O = ir::Opcode;
  switch (op) {
    case O::ADD:       return "add";
    case O::SUB:       return "sub";
    case O::MUL:       return "mul";
    case O::DIV:       return "div";
    case O::MAD:       return "mad";
    case O::REM:       return "rem";
    case O::FADD:      return "add";
    case O::FSUB:      return "sub";
    case O::FMUL:      return "mul";
    case O::FDIV:      return "div";
    case O::FFMA:      return "fma";
    case O::AND:       return "and";
    case O::OR:        return "or";
    case O::XOR:       return "xor";
    case O::NOT:       return "not";
    case O::SHL:       return "shl";
    case O::SHR:       return "shr";
    case O::SETP_EQ:   return "setp.eq";
    case O::SETP_NE:   return "setp.ne";
    case O::SETP_LT:   return "setp.lt";
    case O::SETP_LE:   return "setp.le";
    case O::SETP_GT:   return "setp.gt";
    case O::SETP_GE:   return "setp.ge";
    case O::FSETP_LT:  return "setp.lt";
    case O::FSETP_GT:  return "setp.gt";
    case O::FSETP_EQ:  return "setp.eq";
    case O::LD_GLOBAL: return "ld.global";
    case O::LD_SHARED: return "ld.shared";
    case O::LD_LOCAL:  return "ld.local";
    case O::LD_PARAM:  return "ld.param";
    case O::ST_GLOBAL: return "st.global";
    case O::ST_SHARED: return "st.shared";
    case O::ST_LOCAL:  return "st.local";
    case O::CVT:       return "cvt";
    case O::MOV:       return "mov";
    case O::COPY:      return "mov";
    case O::BARRIER:   return "bar.sync";
    case O::RET:       return "ret";
    default:           return "/* unknown */";
  }
}

// ---------------------------------------------------------------------------
// Module / function prologue
// ---------------------------------------------------------------------------

void PTXEmitter::emitModule(const ir::Module& mod) {
  out_ << ".version 8.0\n"
       << ".target " << sm_ << "\n"
       << ".address_size 64\n\n";
  for (auto& fn : mod.functions)
    emitFunction(*fn);
}

void PTXEmitter::emitPrologue(const ir::Function& fn) {
  out_ << ".visible .entry " << fn.name << "(\n";
  for (size_t i = 0; i < fn.params.size(); ++i) {
    const auto* vr = fn.vregs[fn.params[i]].get();
    out_ << "\t.param ." << vr->type->ptxName()
         << " " << fn.name << "_param_" << i;
    if (i + 1 < fn.params.size()) out_ << ",";
    out_ << "\n";
  }
  out_ << ")\n{\n";

  // Declare register banks
  uint32_t maxR = 0, maxF = 0, maxP = 0, maxRD = 0;
  for (auto& vr : fn.vregs) {
    if (!vr->isAssigned()) continue;
    uint32_t pr = vr->physReg;
    if (vr->type->isPred())         maxP  = std::max(maxP,  pr + 1);
    else if (vr->type->isFloat())   maxF  = std::max(maxF,  pr + 1);
    else if (vr->type->bitWidth() == 64) maxRD = std::max(maxRD, pr + 1);
    else                            maxR  = std::max(maxR,  pr + 1);
  }
  if (maxP)  out_ << "\t.reg .pred   %p<"  << maxP  << ">;\n";
  if (maxF)  out_ << "\t.reg .f32    %f<"  << maxF  << ">;\n";
  if (maxRD) out_ << "\t.reg .s64    %rd<" << maxRD << ">;\n";
  if (maxR)  out_ << "\t.reg .s32    %r<"  << maxR  << ">;\n";
  out_ << "\n";
}

void PTXEmitter::emitEpilogue() {
  out_ << "}\n\n";
}

// ---------------------------------------------------------------------------
// Basic block and instruction emission
// ---------------------------------------------------------------------------

void PTXEmitter::emitFunction(const ir::Function& fn) {
  emitPrologue(fn);
  for (auto& bb : fn.blocks)
    emitBlock(fn, *bb);
  emitEpilogue();
}

void PTXEmitter::emitBlock(const ir::Function& fn, const ir::BasicBlock& bb) {
  out_ << bb.name << ":\n";
  for (auto& instr : bb.instrs)
    emitInstr(fn, *instr);
  out_ << "\n";
}

void PTXEmitter::emitInstr(const ir::Function& fn,
                            const ir::Instruction& instr) {
  using O = ir::Opcode;
  std::ostringstream s;

  auto destName = [&]() {
    return instr.definesDest() ? physRegName(fn, instr.dest) : "";
  };
  auto useName = [&](size_t i) {
    return physRegName(fn, instr.uses[i]);
  };

  switch (instr.opcode) {
    // --- Arithmetic ---
    case O::ADD: case O::SUB: case O::MUL: case O::DIV:
    case O::FADD: case O::FSUB: case O::FMUL: case O::FDIV:
    case O::AND: case O::OR: case O::XOR: case O::SHL: case O::SHR: {
      auto* vr = fn.vregs[instr.dest].get();
      s << opcodeStr(instr.opcode) << "." << vr->type->ptxName()
        << " " << destName() << ", " << useName(0) << ", ";
      if (instr.hasImm) s << instr.imm;
      else              s << useName(1);
      s << ";";
      break;
    }
    case O::MAD: case O::FFMA:
      s << opcodeStr(instr.opcode) << ".lo.s32 "
        << destName() << ", " << useName(0) << ", "
        << useName(1) << ", " << useName(2) << ";";
      break;
    case O::NOT:
      s << "not.b32 " << destName() << ", " << useName(0) << ";";
      break;

    // --- Comparisons ---
    case O::SETP_EQ: case O::SETP_NE: case O::SETP_LT:
    case O::SETP_LE: case O::SETP_GT: case O::SETP_GE:
    case O::FSETP_LT: case O::FSETP_GT: case O::FSETP_EQ: {
      auto* src = fn.vregs[instr.uses[0]].get();
      s << opcodeStr(instr.opcode) << "." << src->type->ptxName()
        << " " << destName() << ", " << useName(0) << ", " << useName(1) << ";";
      break;
    }

    // --- Memory ---
    case O::LD_PARAM:
      s << "ld.param." << fn.vregs[instr.dest]->type->ptxName()
        << " " << destName() << ", [" << instr.label << "];";
      break;
    case O::LD_GLOBAL: case O::LD_SHARED: case O::LD_LOCAL:
      s << opcodeStr(instr.opcode) << "."
        << fn.vregs[instr.dest]->type->ptxName()
        << " " << destName() << ", [" << useName(0) << "];";
      break;
    case O::ST_GLOBAL: case O::ST_SHARED: case O::ST_LOCAL:
      s << opcodeStr(instr.opcode) << "."
        << fn.vregs[instr.uses[0]]->type->ptxName()
        << " [" << useName(0) << "], " << useName(1) << ";";
      break;

    // --- MOV / COPY ---
    case O::MOV: case O::COPY:
      s << "mov." << fn.vregs[instr.dest]->type->ptxName()
        << " " << destName() << ", ";
      if (instr.hasImm) s << instr.imm;
      else              s << useName(0);
      s << ";";
      break;
    case O::CVT: {
      auto* dst = fn.vregs[instr.dest].get();
      auto* src = fn.vregs[instr.uses[0]].get();
      s << "cvt." << dst->type->ptxName() << "." << src->type->ptxName()
        << " " << destName() << ", " << useName(0) << ";";
      break;
    }

    // --- Control flow ---
    case O::BRA:
      s << "bra " << (instr.succBB ? instr.succBB->name : "?") << ";";
      break;
    case O::BRA_PRED:
      s << "@" << useName(0) << " bra "
        << (instr.succBB ? instr.succBB->name : "?") << ";";
      break;
    case O::RET:
      s << "ret;";
      break;
    case O::CALL:
      s << "call " << instr.label << ";";
      break;

    // --- GPU intrinsics ---
    case O::THREAD_IDX_X:
      s << "mov.u32 " << destName() << ", %tid.x;";  break;
    case O::THREAD_IDX_Y:
      s << "mov.u32 " << destName() << ", %tid.y;";  break;
    case O::THREAD_IDX_Z:
      s << "mov.u32 " << destName() << ", %tid.z;";  break;
    case O::BLOCK_IDX_X:
      s << "mov.u32 " << destName() << ", %ctaid.x;"; break;
    case O::BLOCK_IDX_Y:
      s << "mov.u32 " << destName() << ", %ctaid.y;"; break;
    case O::BLOCK_DIM_X:
      s << "mov.u32 " << destName() << ", %ntid.x;";  break;
    case O::BARRIER:
      s << "bar.sync 0;"; break;

    // --- PHI (should have been lowered before emission) ---
    case O::PHI:
      s << "/* PHI %" << instr.dest << " — should be lowered */";
      break;

    default:
      s << "/* unhandled opcode */";
  }
  line(s.str());
}

} // namespace codegen
} // namespace simt

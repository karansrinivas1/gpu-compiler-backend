#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/// SIMT Compiler Backend — Intermediate Representation
///
/// Defines the GPU virtual-register SSA IR used by the register allocator.
/// Modelled after LLVM MIR: instructions operate on unbounded virtual
/// registers; the register allocator maps them to a finite physical file.

namespace simt {
namespace ir {

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

enum class TypeKind : uint8_t {
  Void,
  Pred,               ///< 1-bit predicate (%p registers in PTX)
  S16, S32, S64,
  U16, U32, U64,
  F16, F32, F64,
  B32, B64,
};

class Type {
public:
  explicit Type(TypeKind k) : kind_(k) {}

  TypeKind kind() const { return kind_; }
  bool isPred()  const { return kind_ == TypeKind::Pred; }
  bool isFloat() const { return kind_ >= TypeKind::F16 && kind_ <= TypeKind::F64; }
  bool isInt()   const { return (kind_ >= TypeKind::S16 && kind_ <= TypeKind::U64)
                              || kind_ == TypeKind::B32 || kind_ == TypeKind::B64; }

  unsigned bitWidth() const {
    switch (kind_) {
      case TypeKind::Pred:                                          return 1;
      case TypeKind::S16: case TypeKind::U16: case TypeKind::F16:  return 16;
      case TypeKind::S32: case TypeKind::U32: case TypeKind::F32:
      case TypeKind::B32:                                           return 32;
      case TypeKind::S64: case TypeKind::U64: case TypeKind::F64:
      case TypeKind::B64:                                           return 64;
      default: return 0;
    }
  }

  std::string ptxName() const {
    switch (kind_) {
      case TypeKind::Pred: return "pred";
      case TypeKind::S16:  return "s16";  case TypeKind::S32: return "s32";
      case TypeKind::S64:  return "s64";  case TypeKind::U16: return "u16";
      case TypeKind::U32:  return "u32";  case TypeKind::U64: return "u64";
      case TypeKind::F16:  return "f16";  case TypeKind::F32: return "f32";
      case TypeKind::F64:  return "f64";  case TypeKind::B32: return "b32";
      case TypeKind::B64:  return "b64";
      default:             return "void";
    }
  }

  // Predefined singleton types
  static Type* Void()  { static Type t(TypeKind::Void);  return &t; }
  static Type* Pred()  { static Type t(TypeKind::Pred);  return &t; }
  static Type* S32()   { static Type t(TypeKind::S32);   return &t; }
  static Type* U32()   { static Type t(TypeKind::U32);   return &t; }
  static Type* F32()   { static Type t(TypeKind::F32);   return &t; }
  static Type* F64()   { static Type t(TypeKind::F64);   return &t; }
  static Type* S64()   { static Type t(TypeKind::S64);   return &t; }
  static Type* U64()   { static Type t(TypeKind::U64);   return &t; }

private:
  TypeKind kind_;
};

// ---------------------------------------------------------------------------
// Virtual registers
// ---------------------------------------------------------------------------

using VRegID = uint32_t;
static constexpr VRegID InvalidVReg = UINT32_MAX;

/// A virtual register in the infinite SSA register file.
struct VReg {
  VRegID  id;
  Type*   type;
  /// Physical register assigned by the allocator (InvalidVReg = unassigned/spilled).
  uint32_t physReg{InvalidVReg};
  /// Spill slot index if this vreg was spilled to local memory.
  int32_t  spillSlot{-1};

  bool isAssigned() const { return physReg != InvalidVReg; }
  bool isSpilled()  const { return spillSlot >= 0; }

  std::string name() const {
    return (type->isPred() ? "%p" : "%r") + std::to_string(id);
  }
};

// ---------------------------------------------------------------------------
// Opcodes
// ---------------------------------------------------------------------------

enum class Opcode : uint16_t {
  // Arithmetic
  ADD, SUB, MUL, DIV, MAD, REM,
  // Float arithmetic
  FADD, FSUB, FMUL, FDIV, FFMA,
  // Bitwise
  AND, OR, XOR, NOT, SHL, SHR,
  // Comparisons (result is Pred)
  SETP_EQ, SETP_NE, SETP_LT, SETP_LE, SETP_GT, SETP_GE,
  FSETP_LT, FSETP_GT, FSETP_EQ,
  // Memory
  LD_GLOBAL, LD_SHARED, LD_LOCAL, LD_PARAM,
  ST_GLOBAL, ST_SHARED, ST_LOCAL,
  // Conversions
  CVT, MOV,
  // Control flow
  BRA,          ///< Unconditional branch
  BRA_PRED,     ///< Predicated branch (@%p bra)
  RET,
  CALL,
  // Special GPU intrinsics
  THREAD_IDX_X, THREAD_IDX_Y, THREAD_IDX_Z,
  BLOCK_IDX_X,  BLOCK_IDX_Y,  BLOCK_IDX_Z,
  BLOCK_DIM_X,
  BARRIER,      ///< __syncthreads()
  // Phi (SSA join point)
  PHI,
  // Pseudo
  COPY,         ///< Register copy — eliminated by coalescing
};

// ---------------------------------------------------------------------------
// Instruction
// ---------------------------------------------------------------------------

class BasicBlock;

class Instruction {
public:
  Opcode              opcode;
  VRegID              dest{InvalidVReg};   ///< Defined virtual register (or InvalidVReg)
  std::vector<VRegID> uses;                ///< Used virtual registers
  BasicBlock*         succBB{nullptr};     ///< Branch target (for BRA*)
  std::string         label;               ///< Optional label / parameter name for LD_PARAM
  int64_t             imm{0};              ///< Immediate operand
  bool                hasImm{false};

  /// For PHI nodes: one (pred, vreg) pair per incoming edge
  std::vector<std::pair<BasicBlock*, VRegID>> phiArgs;

  explicit Instruction(Opcode op) : opcode(op) {}

  bool definesDest()  const { return dest != InvalidVReg; }
  bool isBranch()     const { return opcode == Opcode::BRA || opcode == Opcode::BRA_PRED; }
  bool isReturn()     const { return opcode == Opcode::RET; }
  bool isPhi()        const { return opcode == Opcode::PHI; }
  bool isCopy()       const { return opcode == Opcode::COPY; }
  bool isMemLoad()    const { return opcode >= Opcode::LD_GLOBAL && opcode <= Opcode::LD_PARAM; }
  bool isMemStore()   const { return opcode >= Opcode::ST_GLOBAL && opcode <= Opcode::ST_LOCAL; }
};

// ---------------------------------------------------------------------------
// BasicBlock
// ---------------------------------------------------------------------------

class BasicBlock {
public:
  std::string                          name;
  std::vector<std::unique_ptr<Instruction>> instrs;
  std::vector<BasicBlock*>             preds;
  std::vector<BasicBlock*>             succs;
  uint32_t                             index{0}; ///< RPO index

  explicit BasicBlock(std::string n) : name(std::move(n)) {}

  void addInstr(std::unique_ptr<Instruction> i) {
    instrs.push_back(std::move(i));
  }
  void addPred(BasicBlock* bb) { preds.push_back(bb); }
  void addSucc(BasicBlock* bb) { succs.push_back(bb); }

  /// Last instruction (must be terminator for well-formed CFG).
  Instruction* terminator() {
    return instrs.empty() ? nullptr : instrs.back().get();
  }
};

// ---------------------------------------------------------------------------
// Function
// ---------------------------------------------------------------------------

class Function {
public:
  std::string                              name;
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  std::vector<std::unique_ptr<VReg>>       vregs;
  std::vector<VRegID>                      params; ///< Parameter vregs

  explicit Function(std::string n) : name(std::move(n)) {}

  /// Create a new virtual register of the given type.
  VReg* newVReg(Type* ty) {
    auto id = static_cast<VRegID>(vregs.size());
    vregs.push_back(std::make_unique<VReg>(VReg{id, ty}));
    return vregs.back().get();
  }

  VReg* getVReg(VRegID id) {
    assert(id < vregs.size() && "VReg ID out of range");
    return vregs[id].get();
  }

  BasicBlock* newBlock(std::string n) {
    auto idx = static_cast<uint32_t>(blocks.size());
    blocks.push_back(std::make_unique<BasicBlock>(std::move(n)));
    blocks.back()->index = idx;
    return blocks.back().get();
  }

  BasicBlock* entry() {
    assert(!blocks.empty());
    return blocks.front().get();
  }

  /// Number of virtual registers (register pressure upper bound).
  uint32_t numVRegs() const { return static_cast<uint32_t>(vregs.size()); }
};

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

class Module {
public:
  std::string                            name;
  std::vector<std::unique_ptr<Function>> functions;

  explicit Module(std::string n) : name(std::move(n)) {}

  Function* newFunction(std::string n) {
    functions.push_back(std::make_unique<Function>(std::move(n)));
    return functions.back().get();
  }
};

} // namespace ir
} // namespace simt

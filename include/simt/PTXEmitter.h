#pragma once

#include "IR.h"
#include <ostream>
#include <string>

/// PTX Code Emitter
///
/// Translates the IR (after register allocation) into NVIDIA PTX assembly.
/// Assumes all virtual registers have been assigned physical registers or
/// spill slots by the register allocator.

namespace simt {
namespace codegen {

class PTXEmitter {
public:
  /// @param sm  Shader model string, e.g. "sm_86"
  explicit PTXEmitter(std::ostream& out, std::string sm = "sm_86")
      : out_(out), sm_(std::move(sm)) {}

  /// Emit an entire module.
  void emitModule(const ir::Module& mod);

  /// Emit a single kernel function.
  void emitFunction(const ir::Function& fn);

private:
  void emitPrologue(const ir::Function& fn);
  void emitBlock(const ir::Function& fn, const ir::BasicBlock& bb);
  void emitInstr(const ir::Function& fn, const ir::Instruction& instr);
  void emitEpilogue();

  std::string physRegName(const ir::Function& fn, ir::VRegID v) const;
  std::string instrSuffix(const ir::Instruction& instr) const;
  std::string opcodeStr(ir::Opcode op) const;

  std::ostream& out_;
  std::string   sm_;
  unsigned      indent_{1};

  void line(const std::string& s);
  void line();
};

} // namespace codegen
} // namespace simt

/// gpucc — SIMT compiler backend driver
///
/// Demonstrates the full pipeline:
///   IR construction → Liveness analysis → Interference graph →
///   Chaitin-Briggs register allocation → PTX emission
///
/// Example kernel: parallel vector-add  C[i] = A[i] + B[i]

#include "simt/IR.h"
#include "simt/Liveness.h"
#include "simt/InterferenceGraph.h"
#include "simt/GraphColoringRA.h"
#include "simt/PTXEmitter.h"
#include <iostream>

using namespace simt;
using namespace simt::ir;

/// Build IR for:
///   __global__ void vecAdd(float* A, float* B, float* C, int N) {
///       int i = blockIdx.x * blockDim.x + threadIdx.x;
///       if (i < N) C[i] = A[i] + B[i];
///   }
static Function* buildVecAddKernel(Module& mod) {
  auto* fn = mod.newFunction("vecAdd");

  // Parameters: A (u64 ptr), B (u64 ptr), C (u64 ptr), N (s32)
  auto* pA = fn->newVReg(Type::U64()); fn->params.push_back(pA->id);
  auto* pB = fn->newVReg(Type::U64()); fn->params.push_back(pB->id);
  auto* pC = fn->newVReg(Type::U64()); fn->params.push_back(pC->id);
  auto* pN = fn->newVReg(Type::S32()); fn->params.push_back(pN->id);

  // --- entry block ---
  auto* entry = fn->newBlock("entry");
  auto* body  = fn->newBlock("body");
  auto* exit_ = fn->newBlock("exit");
  entry->addSucc(body);  body->addPred(entry);
  entry->addSucc(exit_); exit_->addPred(entry);
  body->addSucc(exit_);  exit_->addPred(body);

  auto emit = [&](BasicBlock* bb, Opcode op) {
    auto i = std::make_unique<Instruction>(op);
    auto* p = i.get();
    bb->addInstr(std::move(i));
    return p;
  };

  // Load parameters
  auto loadParam = [&](BasicBlock* bb, VReg* dest, const std::string& paramName) {
    auto* i  = emit(bb, Opcode::LD_PARAM);
    i->dest  = dest->id;
    i->label = paramName;
  };

  loadParam(entry, pA, "vecAdd_param_0");
  loadParam(entry, pB, "vecAdd_param_1");
  loadParam(entry, pC, "vecAdd_param_2");
  loadParam(entry, pN, "vecAdd_param_3");

  // int i = blockIdx.x * blockDim.x + threadIdx.x
  auto* vBidx  = fn->newVReg(Type::U32());
  auto* vBdim  = fn->newVReg(Type::U32());
  auto* vTidx  = fn->newVReg(Type::U32());
  auto* vBase  = fn->newVReg(Type::U32());
  auto* vi     = fn->newVReg(Type::S32());

  auto* iBidx = emit(entry, Opcode::BLOCK_IDX_X);  iBidx->dest = vBidx->id;
  auto* iBdim = emit(entry, Opcode::BLOCK_DIM_X);  iBdim->dest = vBdim->id;
  auto* iTidx = emit(entry, Opcode::THREAD_IDX_X); iTidx->dest = vTidx->id;

  auto* iMul = emit(entry, Opcode::MUL);
  iMul->dest = vBase->id;
  iMul->uses = {vBidx->id, vBdim->id};

  auto* iAdd = emit(entry, Opcode::ADD);
  iAdd->dest = vi->id;
  iAdd->uses = {vBase->id, vTidx->id};

  // if (i >= N) goto exit
  auto* vPred = fn->newVReg(Type::Pred());
  auto* iSetp = emit(entry, Opcode::SETP_LT);
  iSetp->dest = vPred->id;
  iSetp->uses = {vi->id, pN->id};

  auto* iBra = emit(entry, Opcode::BRA_PRED);
  iBra->uses   = {vPred->id};
  iBra->succBB = body;

  auto* iExit0 = emit(entry, Opcode::BRA);
  iExit0->succBB = exit_;

  // --- body block ---
  // Compute byte offset: off = i * 4  (float = 4 bytes)
  auto* vI64   = fn->newVReg(Type::S64());
  auto* vOff   = fn->newVReg(Type::S64());

  auto* iCvt = emit(body, Opcode::CVT);
  iCvt->dest = vI64->id;
  iCvt->uses = {vi->id};

  auto* iOff = emit(body, Opcode::MUL);
  iOff->dest   = vOff->id;
  iOff->uses   = {vI64->id};
  iOff->imm    = 4;
  iOff->hasImm = true;

  // addr_A = A + off
  auto* vAddrA = fn->newVReg(Type::U64());
  auto* vAddrB = fn->newVReg(Type::U64());
  auto* vAddrC = fn->newVReg(Type::U64());

  auto mkAdd64 = [&](BasicBlock* bb, VReg* dst, VReg* base, VReg* off) {
    auto* i  = emit(bb, Opcode::ADD);
    i->dest  = dst->id;
    i->uses  = {base->id, off->id};
  };
  mkAdd64(body, vAddrA, pA, vOff);
  mkAdd64(body, vAddrB, pB, vOff);
  mkAdd64(body, vAddrC, pC, vOff);

  // va = A[i]; vb = B[i]
  auto* va = fn->newVReg(Type::F32());
  auto* vb = fn->newVReg(Type::F32());
  auto* vc = fn->newVReg(Type::F32());

  auto* iLdA = emit(body, Opcode::LD_GLOBAL);
  iLdA->dest = va->id; iLdA->uses = {vAddrA->id};

  auto* iLdB = emit(body, Opcode::LD_GLOBAL);
  iLdB->dest = vb->id; iLdB->uses = {vAddrB->id};

  // vc = va + vb
  auto* iFAdd = emit(body, Opcode::FADD);
  iFAdd->dest = vc->id;
  iFAdd->uses = {va->id, vb->id};

  // C[i] = vc
  auto* iSt = emit(body, Opcode::ST_GLOBAL);
  iSt->uses = {vAddrC->id, vc->id};

  auto* iBra2 = emit(body, Opcode::BRA);
  iBra2->succBB = exit_;

  // --- exit block ---
  emit(exit_, Opcode::RET);

  return fn;
}

/// Build a simple matrix-multiply inner loop body to stress the allocator
/// with higher register pressure.
static Function* buildMatMulKernel(Module& mod) {
  auto* fn = mod.newFunction("matMul");

  auto* pA = fn->newVReg(Type::U64()); fn->params.push_back(pA->id);
  auto* pB = fn->newVReg(Type::U64()); fn->params.push_back(pB->id);
  auto* pC = fn->newVReg(Type::U64()); fn->params.push_back(pC->id);
  auto* pN = fn->newVReg(Type::S32()); fn->params.push_back(pN->id);

  auto* entry   = fn->newBlock("entry");
  auto* loopHdr = fn->newBlock("loop_header");
  auto* loopBdy = fn->newBlock("loop_body");
  auto* loopEnd = fn->newBlock("loop_exit");

  entry->addSucc(loopHdr); loopHdr->addPred(entry);
  loopHdr->addSucc(loopBdy); loopBdy->addPred(loopHdr);
  loopHdr->addSucc(loopEnd); loopEnd->addPred(loopHdr);
  loopBdy->addSucc(loopHdr); loopHdr->addPred(loopBdy);

  auto emit = [&](BasicBlock* bb, Opcode op) {
    auto i = std::make_unique<Instruction>(op);
    auto* p = i.get();
    bb->addInstr(std::move(i));
    return p;
  };

  // row = blockIdx.y * blockDim.x + threadIdx.y
  // col = blockIdx.x * blockDim.x + threadIdx.x
  auto* vRow = fn->newVReg(Type::S32());
  auto* vCol = fn->newVReg(Type::S32());
  auto* vBx  = fn->newVReg(Type::U32()); emit(entry, Opcode::BLOCK_IDX_X)->dest = vBx->id;
  auto* vBy  = fn->newVReg(Type::U32()); emit(entry, Opcode::BLOCK_IDX_Y)->dest = vBy->id;
  auto* vTx  = fn->newVReg(Type::U32()); emit(entry, Opcode::THREAD_IDX_X)->dest = vTx->id;
  auto* vTy  = fn->newVReg(Type::U32()); emit(entry, Opcode::THREAD_IDX_Y)->dest = vTy->id;
  auto* vBd  = fn->newVReg(Type::U32()); emit(entry, Opcode::BLOCK_DIM_X)->dest = vBd->id;

  auto mkMul = [&](BasicBlock* bb, VReg* dst, VReg* a, VReg* b) {
    auto* i = emit(bb, Opcode::MUL);
    i->dest = dst->id; i->uses = {a->id, b->id};
  };
  auto mkAdd = [&](BasicBlock* bb, VReg* dst, VReg* a, VReg* b) {
    auto* i = emit(bb, Opcode::ADD);
    i->dest = dst->id; i->uses = {a->id, b->id};
  };

  auto* vBxd = fn->newVReg(Type::U32()); mkMul(entry, vBxd, vBx, vBd);
  auto* vByd = fn->newVReg(Type::U32()); mkMul(entry, vByd, vBy, vBd);
  mkAdd(entry, vCol, vBxd, vTx);
  mkAdd(entry, vRow, vByd, vTy);

  // sum = 0.0f
  auto* vSum = fn->newVReg(Type::F32());
  auto* iMov = emit(entry, Opcode::MOV);
  iMov->dest = vSum->id; iMov->imm = 0; iMov->hasImm = true;

  // k = 0
  auto* vK = fn->newVReg(Type::S32());
  auto* iK0 = emit(entry, Opcode::MOV);
  iK0->dest = vK->id; iK0->imm = 0; iK0->hasImm = true;

  emit(entry, Opcode::BRA)->succBB = loopHdr;

  // PHI for sum and k in loop header
  auto* vSumPhi = fn->newVReg(Type::F32());
  auto* vKPhi   = fn->newVReg(Type::S32());

  auto* phiSum = emit(loopHdr, Opcode::PHI);
  phiSum->dest = vSumPhi->id;
  phiSum->phiArgs = {{entry, vSum->id}, {loopBdy, ir::InvalidVReg}}; // body updated below

  auto* phiK = emit(loopHdr, Opcode::PHI);
  phiK->dest = vKPhi->id;
  phiK->phiArgs = {{entry, vK->id}, {loopBdy, ir::InvalidVReg}};

  // loop cond: k < N
  auto* vPred = fn->newVReg(Type::Pred());
  auto* iSetp = emit(loopHdr, Opcode::SETP_LT);
  iSetp->dest = vPred->id; iSetp->uses = {vKPhi->id, pN->id};
  emit(loopHdr, Opcode::BRA_PRED)->succBB = loopBdy;
  auto* iBraExit = emit(loopHdr, Opcode::BRA_PRED);
  iBraExit->succBB = loopEnd;

  // loop body: sum += A[row*N+k] * B[k*N+col]
  auto* vRN = fn->newVReg(Type::S32()); mkMul(loopBdy, vRN, vRow, pN);
  auto* vRK = fn->newVReg(Type::S32()); mkAdd(loopBdy, vRK, vRN, vKPhi);
  auto* vCK = fn->newVReg(Type::S32());
  auto* vKN = fn->newVReg(Type::S32()); mkMul(loopBdy, vKN, vKPhi, pN);
  mkAdd(loopBdy, vCK, vKN, vCol);

  // loads
  auto cvt = [&](BasicBlock* bb, VReg* dst, VReg* src) {
    auto* i = emit(bb, Opcode::CVT); i->dest = dst->id; i->uses = {src->id};
  };
  auto* vRK64 = fn->newVReg(Type::S64()); cvt(loopBdy, vRK64, vRK);
  auto* vCK64 = fn->newVReg(Type::S64()); cvt(loopBdy, vCK64, vCK);
  auto* vOffA = fn->newVReg(Type::S64());
  auto* iMulA = emit(loopBdy, Opcode::MUL);
  iMulA->dest = vOffA->id; iMulA->uses = {vRK64->id}; iMulA->imm = 4; iMulA->hasImm = true;
  auto* vOffB = fn->newVReg(Type::S64());
  auto* iMulB = emit(loopBdy, Opcode::MUL);
  iMulB->dest = vOffB->id; iMulB->uses = {vCK64->id}; iMulB->imm = 4; iMulB->hasImm = true;

  auto* vAddrA = fn->newVReg(Type::U64()); mkAdd(loopBdy, vAddrA, pA, vOffA);
  auto* vAddrB = fn->newVReg(Type::U64()); mkAdd(loopBdy, vAddrB, pB, vOffB);

  auto* vA  = fn->newVReg(Type::F32());
  auto* vB  = fn->newVReg(Type::F32());
  emit(loopBdy, Opcode::LD_GLOBAL)->dest = vA->id;
  loopBdy->instrs.back()->uses = {vAddrA->id};
  emit(loopBdy, Opcode::LD_GLOBAL)->dest = vB->id;
  loopBdy->instrs.back()->uses = {vAddrB->id};

  // sum += a * b  (fused multiply-add)
  auto* vNewSum = fn->newVReg(Type::F32());
  auto* iFma = emit(loopBdy, Opcode::FFMA);
  iFma->dest = vNewSum->id; iFma->uses = {vA->id, vB->id, vSumPhi->id};

  // k++
  auto* vNewK = fn->newVReg(Type::S32());
  auto* iKInc = emit(loopBdy, Opcode::ADD);
  iKInc->dest = vNewK->id; iKInc->uses = {vKPhi->id}; iKInc->imm = 1; iKInc->hasImm = true;

  // Fix PHI back-edges
  phiSum->phiArgs[1] = {loopBdy, vNewSum->id};
  phiK->phiArgs[1]   = {loopBdy, vNewK->id};

  emit(loopBdy, Opcode::BRA)->succBB = loopHdr;

  // loop exit: store C[row*N+col] = sum
  auto* vOutIdx = fn->newVReg(Type::S32()); mkMul(loopEnd, vOutIdx, vRow, pN);
  auto* vOutRK  = fn->newVReg(Type::S32()); mkAdd(loopEnd, vOutRK, vOutIdx, vCol);
  auto* vOut64  = fn->newVReg(Type::S64()); cvt(loopEnd, vOut64, vOutRK);
  auto* vOutOff = fn->newVReg(Type::S64());
  auto* iMulOut = emit(loopEnd, Opcode::MUL);
  iMulOut->dest = vOutOff->id; iMulOut->uses = {vOut64->id}; iMulOut->imm = 4; iMulOut->hasImm = true;
  auto* vAddrC = fn->newVReg(Type::U64()); mkAdd(loopEnd, vAddrC, pC, vOutOff);

  auto* iSt = emit(loopEnd, Opcode::ST_GLOBAL);
  iSt->uses = {vAddrC->id, vSumPhi->id};

  emit(loopEnd, Opcode::RET);

  return fn;
}

int main() {
  auto mod = Module("kernels");
  auto rf = regalloc::TargetRegisterFile::sm86();

  // --- vecAdd ---
  {
    auto* fn = buildVecAddKernel(mod);
    std::cout << "=== vecAdd (" << fn->numVRegs() << " virtual regs) ===\n";

    analysis::LivenessAnalysis la(*fn);
    la.run();
    std::cout << "\n--- Liveness ---\n";
    la.dump();

    auto ig = analysis::InterferenceGraph::build(*fn, la);
    std::cout << "\n--- Interference Graph ---\n";
    ig.dump();

    regalloc::GraphColoringRA alloc(*fn, rf);
    auto result = alloc.run();
    std::cout << "\n--- Allocation ---\n";
    result.dump();

    std::cout << "\n--- PTX Output ---\n";
    codegen::PTXEmitter emitter(std::cout);
    emitter.emitFunction(*fn);
  }

  // --- matMul ---
  {
    auto* fn = buildMatMulKernel(mod);
    std::cout << "\n=== matMul (" << fn->numVRegs() << " virtual regs) ===\n";

    regalloc::GraphColoringRA alloc(*fn, rf);
    auto result = alloc.run();
    std::cout << "--- Allocation ---\n";
    result.dump();

    std::cout << "\n--- PTX Output ---\n";
    codegen::PTXEmitter emitter(std::cout);
    emitter.emitFunction(*fn);
  }

  return 0;
}

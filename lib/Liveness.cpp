#include "simt/Liveness.h"
#include <algorithm>
#include <iostream>
#include <queue>

namespace simt {
namespace analysis {

// ---------------------------------------------------------------------------
// LivenessAnalysis
// ---------------------------------------------------------------------------

void LivenessAnalysis::run() {
  info_.assign(fn_.blocks.size(), BlockLiveness{});
  computeLocalSets();
  // Iterate backward to fixed point
  bool changed = true;
  while (changed) changed = propagate();
}

/// Compute per-block Use and Def sets by scanning instructions in order.
void LivenessAnalysis::computeLocalSets() {
  for (auto& bb : fn_.blocks) {
    auto& info = info_[bb->index];
    for (auto& instr : bb->instrs) {
      // PHI uses come from predecessors — handled separately during propagation.
      if (!instr->isPhi()) {
        for (auto vid : instr->uses) {
          if (vid != ir::InvalidVReg && info.def.find(vid) == info.def.end())
            info.use.insert(vid);
        }
      }
      if (instr->definesDest())
        info.def.insert(instr->dest);
    }
  }
}

/// One backward sweep through all blocks.  Returns true if any liveIn/liveOut
/// set changed.
bool LivenessAnalysis::propagate() {
  bool changed = false;

  // Iterate blocks in reverse order (approximates RPO-backward traversal).
  for (int i = static_cast<int>(fn_.blocks.size()) - 1; i >= 0; --i) {
    auto* bb     = fn_.blocks[i].get();
    auto& info   = info_[bb->index];

    // LiveOut(B) = ∪ LiveIn(S) for each successor S
    VRegSet newOut;
    for (auto* succ : bb->succs) {
      const auto& si = info_[succ->index];
      newOut.insert(si.liveIn.begin(), si.liveIn.end());

      // PHI arguments in the successor that come from this block are live-out here.
      for (auto& instr : succ->instrs) {
        if (!instr->isPhi()) break; // PHIs are always first
        for (auto& [pred, vid] : instr->phiArgs) {
          if (pred == bb && vid != ir::InvalidVReg)
            newOut.insert(vid);
        }
      }
    }

    // LiveIn(B) = Use(B) ∪ (LiveOut(B) − Def(B))
    VRegSet newIn = info.use;
    for (auto v : newOut)
      if (info.def.find(v) == info.def.end())
        newIn.insert(v);

    if (newIn != info.liveIn || newOut != info.liveOut) {
      info.liveIn  = std::move(newIn);
      info.liveOut = std::move(newOut);
      changed = true;
    }
  }
  return changed;
}

VRegSet LivenessAnalysis::liveBeforeInstr(const ir::BasicBlock* bb,
                                           size_t idx) const {
  const auto& info = info_[bb->index];
  // Start from liveOut, scan backward from end to idx.
  VRegSet live = info.liveOut;

  for (int j = static_cast<int>(bb->instrs.size()) - 1;
       j >= static_cast<int>(idx); --j) {
    const auto& instr = *bb->instrs[j];
    if (instr.definesDest()) live.erase(instr.dest);
    for (auto v : instr.uses)
      if (v != ir::InvalidVReg) live.insert(v);
  }
  return live;
}

void LivenessAnalysis::dump() const {
  for (const auto& bb : fn_.blocks) {
    const auto& info = info_[bb->index];
    std::cout << "[" << bb->name << "]\n";
    std::cout << "  use:     ";
    for (auto v : info.use)     std::cout << "%" << v << " ";
    std::cout << "\n  def:     ";
    for (auto v : info.def)     std::cout << "%" << v << " ";
    std::cout << "\n  liveIn:  ";
    for (auto v : info.liveIn)  std::cout << "%" << v << " ";
    std::cout << "\n  liveOut: ";
    for (auto v : info.liveOut) std::cout << "%" << v << " ";
    std::cout << "\n";
  }
}

// ---------------------------------------------------------------------------
// Live Intervals
// ---------------------------------------------------------------------------

/// Assign a linear program point to every instruction.
/// Block b gets instruction indices [2*b->index*W, 2*(b->index*W + |instrs|)).
/// The factor 2 creates gaps for inserting spill/reload pseudos.
std::vector<LiveInterval> buildLiveIntervals(ir::Function&          fn,
                                              const LivenessAnalysis& la) {
  const uint32_t W = 64; // max instrs per block (for spacing)
  std::vector<LiveInterval> intervals(fn.numVRegs());
  for (uint32_t i = 0; i < fn.numVRegs(); ++i)
    intervals[i].vreg = static_cast<ir::VRegID>(i);

  for (auto& bb : fn.blocks) {
    const uint32_t base = bb->index * W * 2;

    // Any vreg live-in to the block is live from the block start.
    for (auto vid : la.blockInfo(bb.get()).liveIn)
      intervals[vid].addRange(base, base + 2);

    for (size_t idx = 0; idx < bb->instrs.size(); ++idx) {
      const auto& instr = *bb->instrs[idx];
      const uint32_t pt = base + static_cast<uint32_t>(idx) * 2;

      // Uses: extend live range to this point
      for (auto v : instr.uses)
        if (v != ir::InvalidVReg)
          intervals[v].addRange(pt, pt + 1);

      // Def: starts a new live range from this point
      if (instr.definesDest())
        intervals[instr.dest].addRange(pt, pt + 2);
    }

    // Vregs live-out: extend to the end of the block
    const uint32_t end = base + static_cast<uint32_t>(bb->instrs.size()) * 2;
    for (auto vid : la.blockInfo(bb.get()).liveOut)
      intervals[vid].addRange(base, end);
  }

  // Merge overlapping/adjacent ranges within each interval
  for (auto& iv : intervals) {
    if (iv.ranges.size() < 2) continue;
    std::sort(iv.ranges.begin(), iv.ranges.end(),
              [](const LiveRange& a, const LiveRange& b) {
                return a.start < b.start;
              });
    std::vector<LiveRange> merged;
    merged.push_back(iv.ranges[0]);
    for (size_t i = 1; i < iv.ranges.size(); ++i) {
      if (iv.ranges[i].start <= merged.back().end)
        merged.back().end = std::max(merged.back().end, iv.ranges[i].end);
      else
        merged.push_back(iv.ranges[i]);
    }
    iv.ranges = std::move(merged);
  }
  return intervals;
}

} // namespace analysis
} // namespace simt

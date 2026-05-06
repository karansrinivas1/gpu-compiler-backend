#pragma once

#include "IR.h"
#include <unordered_set>
#include <vector>

/// Liveness Analysis
///
/// Classic backward dataflow analysis over the CFG:
///   LiveOut(B) = ∪ LiveIn(S)  for every successor S of B
///   LiveIn(B)  = Use(B) ∪ (LiveOut(B) − Def(B))
///
/// Iterated to a fixed point.  Results are used by the interference-graph
/// builder to insert edges between simultaneously live virtual registers.

namespace simt {
namespace analysis {

using VRegSet = std::unordered_set<ir::VRegID>;

struct BlockLiveness {
  VRegSet use;     ///< Upward-exposed uses in this block (read before any def)
  VRegSet def;     ///< Definitions in this block
  VRegSet liveIn;  ///< Live at block entry
  VRegSet liveOut; ///< Live at block exit
};

class LivenessAnalysis {
public:
  explicit LivenessAnalysis(ir::Function& fn) : fn_(fn) {}

  /// Run the analysis.  Call before querying any results.
  void run();

  const BlockLiveness& blockInfo(const ir::BasicBlock* bb) const {
    return info_.at(bb->index);
  }

  /// Returns true if vreg is live at the entry of `bb`.
  bool isLiveIn(const ir::BasicBlock* bb, ir::VRegID vreg) const {
    const auto& li = info_.at(bb->index);
    return li.liveIn.count(vreg) > 0;
  }

  /// Returns true if vreg is live at the exit of `bb`.
  bool isLiveOut(const ir::BasicBlock* bb, ir::VRegID vreg) const {
    const auto& li = info_.at(bb->index);
    return li.liveOut.count(vreg) > 0;
  }

  /// Compute live set at the program point *before* instruction `idx` in `bb`.
  VRegSet liveBeforeInstr(const ir::BasicBlock* bb, size_t idx) const;

  void dump() const;

private:
  void computeLocalSets();
  bool propagate();           ///< One backward pass; returns true if changed

  ir::Function&               fn_;
  std::vector<BlockLiveness>  info_;
};

// ---------------------------------------------------------------------------
// Live Intervals
// ---------------------------------------------------------------------------

/// A contiguous range [start, end) in a linearised instruction numbering.
/// Instructions are numbered 2*block_index + 2*instr_index so that we can
/// insert half-open ranges cleanly.
struct LiveRange {
  uint32_t start;
  uint32_t end;   ///< Exclusive

  bool overlaps(const LiveRange& o) const {
    return start < o.end && o.start < end;
  }
};

/// Full live interval for one virtual register (may be split into multiple
/// LiveRange segments after coalescing or splitting).
struct LiveInterval {
  ir::VRegID            vreg;
  std::vector<LiveRange> ranges;

  void addRange(uint32_t s, uint32_t e) {
    ranges.push_back({s, e});
  }

  bool overlaps(const LiveInterval& o) const {
    for (const auto& a : ranges)
      for (const auto& b : o.ranges)
        if (a.overlaps(b)) return true;
    return false;
  }

  uint32_t start() const { return ranges.empty() ? 0 : ranges.front().start; }
  uint32_t end()   const { return ranges.empty() ? 0 : ranges.back().end; }
};

/// Build live intervals from liveness analysis results.
/// Produces one LiveInterval per virtual register.
std::vector<LiveInterval> buildLiveIntervals(ir::Function&          fn,
                                              const LivenessAnalysis& la);

} // namespace analysis
} // namespace simt

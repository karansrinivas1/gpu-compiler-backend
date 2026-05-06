#include "simt/InterferenceGraph.h"
#include <iostream>

namespace simt {
namespace analysis {

InterferenceGraph::InterferenceGraph(uint32_t numVRegs)
    : numVRegs_(numVRegs),
      adj_(numVRegs),
      affinity_(numVRegs) {}

void InterferenceGraph::addEdge(ir::VRegID u, ir::VRegID v) {
  if (u == v) return;
  if (adj_[u].insert(v).second) {
    adj_[v].insert(u);
    ++numEdges_;
  }
}

void InterferenceGraph::addAffinityEdge(ir::VRegID u, ir::VRegID v) {
  if (u == v) return;
  affinity_[u].insert(v);
  affinity_[v].insert(u);
}

bool InterferenceGraph::interferes(ir::VRegID u, ir::VRegID v) const {
  return adj_[u].count(v) > 0;
}

bool InterferenceGraph::areAffinity(ir::VRegID u, ir::VRegID v) const {
  return affinity_[u].count(v) > 0;
}

const std::unordered_set<ir::VRegID>&
InterferenceGraph::neighbors(ir::VRegID u) const {
  return adj_[u];
}

// ---------------------------------------------------------------------------
// Build from liveness analysis
// ---------------------------------------------------------------------------

/// For each instruction that defines a vreg d, insert edges d→v for every
/// vreg v that is live at that definition point (i.e. simultaneously live).
InterferenceGraph InterferenceGraph::build(ir::Function&          fn,
                                            const LivenessAnalysis& la) {
  InterferenceGraph ig(fn.numVRegs());

  for (auto& bb : fn.blocks) {
    // Compute live set scanning backward through the block
    VRegSet live = la.blockInfo(bb.get()).liveOut;

    for (int i = static_cast<int>(bb->instrs.size()) - 1; i >= 0; --i) {
      const auto& instr = *bb->instrs[i];

      if (instr.isPhi()) {
        // PHI defs interfere with the live set at block entry (excluding
        // other PHI defs of the same block — they are parallel assignments).
        if (instr.definesDest()) {
          for (auto v : live)
            ig.addEdge(instr.dest, v);
        }
        // PHI uses are live-in from each predecessor — not added here.
        continue;
      }

      // For regular instructions: the defined vreg interferes with all
      // currently live vregs.
      if (instr.definesDest()) {
        for (auto v : live)
          ig.addEdge(instr.dest, v);
        live.erase(instr.dest);
      }

      // Uses become live above the instruction.
      for (auto v : instr.uses)
        if (v != ir::InvalidVReg) live.insert(v);

      // Track copy-related pairs for coalescing hints.
      if (instr.isCopy() && instr.definesDest() && !instr.uses.empty())
        ig.addAffinityEdge(instr.dest, instr.uses[0]);
    }
  }
  return ig;
}

InterferenceGraph InterferenceGraph::buildFromIntervals(
    const std::vector<LiveInterval>& intervals, uint32_t numVRegs) {
  InterferenceGraph ig(numVRegs);
  for (size_t i = 0; i < intervals.size(); ++i)
    for (size_t j = i + 1; j < intervals.size(); ++j)
      if (intervals[i].overlaps(intervals[j]))
        ig.addEdge(intervals[i].vreg, intervals[j].vreg);
  return ig;
}

void InterferenceGraph::coalesce(ir::VRegID keep, ir::VRegID merge) {
  // Redirect all edges from merge to keep (skip existing edges to keep)
  for (auto v : adj_[merge]) {
    if (v == keep) continue;
    adj_[v].erase(merge);
    adj_[v].insert(keep);
    adj_[keep].insert(v);
  }
  // Remove the edge between keep and merge (they're now the same node)
  adj_[keep].erase(merge);
  adj_[merge].clear();

  // Merge affinity edges
  for (auto v : affinity_[merge]) {
    if (v == keep) continue;
    affinity_[keep].insert(v);
    affinity_[v].erase(merge);
    affinity_[v].insert(keep);
  }
  affinity_[merge].clear();
}

void InterferenceGraph::dump() const {
  std::cout << "Interference Graph (" << numVRegs_ << " vregs, "
            << numEdges_ << " edges):\n";
  for (uint32_t i = 0; i < numVRegs_; ++i) {
    if (adj_[i].empty()) continue;
    std::cout << "  %" << i << " interferes with:";
    for (auto v : adj_[i]) std::cout << " %" << v;
    std::cout << "\n";
  }
}

} // namespace analysis
} // namespace simt

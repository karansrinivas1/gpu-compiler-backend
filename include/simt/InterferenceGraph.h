#pragma once

#include "IR.h"
#include "Liveness.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>

/// Register Interference Graph (RIG)
///
/// Vertices  : virtual registers
/// Edges     : two vregs interfere if they are simultaneously live at any
///             program point — i.e. they cannot share a physical register.
///
/// Additionally tracks affinity edges (copy-related vregs) used by the
/// coalescing phase of the Chaitin-Briggs allocator.

namespace simt {
namespace analysis {

class InterferenceGraph {
public:
  explicit InterferenceGraph(uint32_t numVRegs);

  /// Add an interference (conflict) edge between two virtual registers.
  void addEdge(ir::VRegID u, ir::VRegID v);

  /// Add a copy-affinity edge (these vregs are copy-related and benefit
  /// from coalescing but do not necessarily conflict).
  void addAffinityEdge(ir::VRegID u, ir::VRegID v);

  bool interferes(ir::VRegID u, ir::VRegID v) const;
  bool areAffinity(ir::VRegID u, ir::VRegID v) const;

  const std::unordered_set<ir::VRegID>& neighbors(ir::VRegID u) const;
  const std::unordered_set<ir::VRegID>& affinity(ir::VRegID u) const {
    return affinity_[u];
  }

  uint32_t degree(ir::VRegID u) const {
    return static_cast<uint32_t>(neighbors(u).size());
  }

  uint32_t numVRegs()  const { return numVRegs_; }
  uint64_t numEdges()  const { return numEdges_; }

  /// Build from liveness analysis: insert an edge for every pair of vregs
  /// that are simultaneously live at any program point.
  static InterferenceGraph build(ir::Function&          fn,
                                 const LivenessAnalysis& la);

  /// Same but based on pre-computed live intervals (faster for large functions).
  static InterferenceGraph buildFromIntervals(
      const std::vector<LiveInterval>& intervals, uint32_t numVRegs);

  void dump() const;

  // Coalescing: merge v into u (contract edge u–v in the graph).
  // All edges formerly incident to v are redirected to u.
  void coalesce(ir::VRegID keep, ir::VRegID merge);

private:
  uint32_t                                          numVRegs_;
  uint64_t                                          numEdges_{0};
  std::vector<std::unordered_set<ir::VRegID>>       adj_;
  std::vector<std::unordered_set<ir::VRegID>>       affinity_;
};

} // namespace analysis
} // namespace simt

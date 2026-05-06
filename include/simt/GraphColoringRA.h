#pragma once

#include "IR.h"
#include "Liveness.h"
#include "InterferenceGraph.h"
#include <functional>
#include <optional>
#include <stack>
#include <unordered_map>
#include <vector>

/// Chaitin-Briggs Graph-Coloring Register Allocator
///
/// Implements the classical Chaitin-Briggs algorithm extended with
/// SIMT-specific occupancy awareness:
///
///   Phase 1 — Build    : liveness + interference graph
///   Phase 2 — Simplify : push low-degree (<K) nodes onto color stack
///   Phase 3 — Coalesce : conservative coalescing of copy-related nodes
///   Phase 4 — Freeze   : give up coalescing one low-degree node if stuck
///   Phase 5 — Spill    : select a potential spill (high-degree node)
///   Phase 6 — Select   : pop stack and assign colours; spill if necessary
///
/// K (number of colours) is derived from:
///   K = total_physical_registers / bytes_per_register
///
/// The allocator is intentionally parameterised so tests can inject small K
/// values to force spilling.

namespace simt {
namespace regalloc {

// ---------------------------------------------------------------------------
// Target description — GPU register file
// ---------------------------------------------------------------------------

struct TargetRegisterFile {
  uint32_t numIntRegs;    ///< e.g. 255 for sm_86
  uint32_t numFloatRegs;  ///< Typically same pool in PTX
  uint32_t numPredRegs;   ///< e.g. 7 for sm_86

  // SIMT occupancy model:
  //   warpsPerSM = regsPerSM / (regsPerThread * warpSize)
  uint32_t regsPerSM;    ///< Total register file per SM (e.g. 65536 for sm_86)
  uint32_t warpSize;     ///< 32 threads per warp

  /// Returns K (coloring count) for registers of the given type.
  uint32_t numColors(ir::TypeKind tk) const {
    return (tk == ir::TypeKind::Pred) ? numPredRegs : numIntRegs;
  }

  /// Estimated active warps per SM given registers-per-thread.
  uint32_t estimateOccupancy(uint32_t regsPerThread) const {
    if (regsPerThread == 0) return 0;
    return regsPerSM / (regsPerThread * warpSize);
  }

  /// sm_86 (Ampere) defaults
  static TargetRegisterFile sm86() {
    return {255, 255, 7, 65536, 32};
  }
};

// ---------------------------------------------------------------------------
// Spill cost model
// ---------------------------------------------------------------------------

struct SpillInfo {
  ir::VRegID vreg;
  float      cost;        ///< Estimated spill cost (uses * loop_depth)
  int32_t    spillSlot;   ///< Assigned local memory offset (bytes)
};

// ---------------------------------------------------------------------------
// Allocator result
// ---------------------------------------------------------------------------

struct AllocResult {
  bool     success;                ///< False if spilling was needed (re-run required)
  uint32_t maxLiveAtAnyPoint;      ///< High-water mark of simultaneously live vregs
  uint32_t regsUsed;               ///< Physical registers consumed
  uint32_t estimatedOccupancy;     ///< Warps per SM using occupancy model
  std::vector<SpillInfo> spills;   ///< Spilled vregs (empty on success)

  void dump() const;
};

// ---------------------------------------------------------------------------
// Graph-coloring register allocator
// ---------------------------------------------------------------------------

class GraphColoringRA {
public:
  explicit GraphColoringRA(ir::Function&              fn,
                           const TargetRegisterFile&  rf)
      : fn_(fn), rf_(rf) {}

  /// Run the full allocator pipeline.
  AllocResult run();

  /// Allow injecting a custom spill-cost function for testing.
  using SpillCostFn = std::function<float(ir::VRegID)>;
  void setSpillCostFn(SpillCostFn fn) { spillCostFn_ = std::move(fn); }

private:
  // Pipeline phases
  void buildPhase();
  void simplifyPhase();
  bool coalescePhase();   ///< Returns true if any coalescing happened
  bool freezePhase();     ///< Returns true if a node was frozen
  ir::VRegID spillPhase();///< Returns the spill candidate
  void selectPhase(std::vector<SpillInfo>& spills);

  // Helpers
  bool isLowDegree(ir::VRegID v) const;
  float spillCost(ir::VRegID v) const;
  void  insertSpillCode(const std::vector<SpillInfo>& spills);
  uint32_t kForVReg(ir::VRegID v) const;

  ir::Function&                          fn_;
  const TargetRegisterFile&              rf_;
  SpillCostFn                            spillCostFn_;

  // Built during buildPhase
  std::unique_ptr<analysis::LivenessAnalysis>  la_;
  std::unique_ptr<analysis::InterferenceGraph> ig_;

  // Work-lists
  std::vector<ir::VRegID>  simplifyWorklist_;   ///< Low-degree non-copy-related
  std::vector<ir::VRegID>  freezeWorklist_;     ///< Low-degree copy-related
  std::vector<ir::VRegID>  spillWorklist_;      ///< High-degree
  std::stack<ir::VRegID>   colorStack_;         ///< For Select phase

  // Coalescing union-find
  std::unordered_map<ir::VRegID, ir::VRegID>   coalesceParent_; ///< find root
  ir::VRegID findRoot(ir::VRegID v) const;
  void        unionCoalesce(ir::VRegID u, ir::VRegID v);

  // Colors assigned in Select phase (physical register index)
  std::unordered_map<ir::VRegID, uint32_t>   color_;

  // Nodes removed from graph during Simplify
  std::unordered_set<ir::VRegID>             removed_;

  int32_t nextSpillSlot_{0}; ///< Next available local-memory byte offset
};

} // namespace regalloc
} // namespace simt

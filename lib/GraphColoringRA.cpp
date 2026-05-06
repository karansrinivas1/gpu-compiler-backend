#include "simt/GraphColoringRA.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>

namespace simt {
namespace regalloc {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

ir::VRegID GraphColoringRA::findRoot(ir::VRegID v) const {
  // Path-compressed find
  while (coalesceParent_.count(v) && coalesceParent_.at(v) != v)
    v = coalesceParent_.at(v);
  return v;
}

void GraphColoringRA::unionCoalesce(ir::VRegID u, ir::VRegID v) {
  coalesceParent_[findRoot(v)] = findRoot(u);
}

uint32_t GraphColoringRA::kForVReg(ir::VRegID v) const {
  auto* vr = fn_.getVReg(v);
  return rf_.numColors(vr->type->kind());
}

bool GraphColoringRA::isLowDegree(ir::VRegID v) const {
  return ig_->degree(v) < kForVReg(v);
}

float GraphColoringRA::spillCost(ir::VRegID v) const {
  if (spillCostFn_) return spillCostFn_(v);
  // Default heuristic: use-count / degree (Bernstein's metric)
  float uses = 0.0f;
  for (auto& bb : fn_.blocks) {
    // Simple loop-depth proxy: blocks with 'loop' in name get weight 10x
    float weight = (bb->name.find("loop") != std::string::npos) ? 10.0f : 1.0f;
    for (auto& instr : bb->instrs) {
      if (instr->definesDest() && instr->dest == v) uses += weight;
      for (auto vid : instr->uses)
        if (vid == v) uses += weight;
    }
  }
  uint32_t deg = ig_->degree(v);
  return (deg == 0) ? std::numeric_limits<float>::max() : uses / deg;
}

// ---------------------------------------------------------------------------
// Phase 1 — Build
// ---------------------------------------------------------------------------

void GraphColoringRA::buildPhase() {
  la_ = std::make_unique<analysis::LivenessAnalysis>(fn_);
  la_->run();

  ig_ = std::make_unique<analysis::InterferenceGraph>(
      analysis::InterferenceGraph::build(fn_, *la_));

  // Initialise coalesceParent_ (identity)
  for (uint32_t i = 0; i < fn_.numVRegs(); ++i)
    coalesceParent_[i] = i;

  // Initialise work-lists
  simplifyWorklist_.clear();
  freezeWorklist_.clear();
  spillWorklist_.clear();

  for (uint32_t i = 0; i < fn_.numVRegs(); ++i) {
    auto v = static_cast<ir::VRegID>(i);
    if (isLowDegree(v)) {
      if (!ig_->affinity(v).empty())
        freezeWorklist_.push_back(v);
      else
        simplifyWorklist_.push_back(v);
    } else {
      spillWorklist_.push_back(v);
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 2 — Simplify
// ---------------------------------------------------------------------------

void GraphColoringRA::simplifyPhase() {
  while (!simplifyWorklist_.empty()) {
    auto v = simplifyWorklist_.back();
    simplifyWorklist_.pop_back();
    colorStack_.push(v);
    removed_.insert(v);

    // Decrement degree of neighbours; some may move to simplifyWorklist
    for (auto nb : ig_->neighbors(v)) {
      if (removed_.count(nb)) continue;
      if (isLowDegree(nb)) {
        // Move from spillWorklist to simplifyWorklist (or freeze)
        auto& sw = spillWorklist_;
        sw.erase(std::remove(sw.begin(), sw.end(), nb), sw.end());
        simplifyWorklist_.push_back(nb);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 3 — Coalesce (conservative Briggs criterion)
// ---------------------------------------------------------------------------

bool GraphColoringRA::coalescePhase() {
  bool coalesced = false;
  for (auto& bb : fn_.blocks) {
    for (auto& instr : bb->instrs) {
      if (!instr->isCopy()) continue;
      if (!instr->definesDest() || instr->uses.empty()) continue;
      auto u = findRoot(instr->dest);
      auto v = findRoot(instr->uses[0]);
      if (u == v) continue;                        // Already same node
      if (ig_->interferes(u, v)) continue;         // Can't coalesce conflicting nodes

      // Briggs criterion: # neighbours of (u ∪ v) with degree >= K < K
      std::unordered_set<ir::VRegID> combined;
      for (auto nb : ig_->neighbors(u)) combined.insert(nb);
      for (auto nb : ig_->neighbors(v)) combined.insert(nb);
      uint32_t K = kForVReg(u);
      uint32_t highDeg = 0;
      for (auto nb : combined)
        if (!removed_.count(nb) && ig_->degree(nb) >= K) ++highDeg;

      if (highDeg < K) {
        ig_->coalesce(u, v);
        unionCoalesce(u, v);
        coalesced = true;
      }
    }
  }
  return coalesced;
}

// ---------------------------------------------------------------------------
// Phase 4 — Freeze
// ---------------------------------------------------------------------------

bool GraphColoringRA::freezePhase() {
  if (freezeWorklist_.empty()) return false;
  auto v = freezeWorklist_.back();
  freezeWorklist_.pop_back();
  simplifyWorklist_.push_back(v);
  return true;
}

// ---------------------------------------------------------------------------
// Phase 5 — Spill selection
// ---------------------------------------------------------------------------

ir::VRegID GraphColoringRA::spillPhase() {
  assert(!spillWorklist_.empty());
  // Choose spill candidate with lowest cost/degree ratio (best to spill cheap,
  // heavily-interfering nodes)
  ir::VRegID best = ir::InvalidVReg;
  float     bestCost = std::numeric_limits<float>::max();
  for (auto v : spillWorklist_) {
    float c = spillCost(v);
    if (c < bestCost) { bestCost = c; best = v; }
  }
  auto& sw = spillWorklist_;
  sw.erase(std::remove(sw.begin(), sw.end(), best), sw.end());
  simplifyWorklist_.push_back(best);
  return best;
}

// ---------------------------------------------------------------------------
// Phase 6 — Select
// ---------------------------------------------------------------------------

void GraphColoringRA::selectPhase(std::vector<SpillInfo>& spills) {
  while (!colorStack_.empty()) {
    auto v = colorStack_.top();
    colorStack_.pop();
    removed_.erase(v);

    uint32_t K = kForVReg(v);

    // Find a colour not used by any neighbour
    std::vector<bool> usedColor(K, false);
    for (auto nb : ig_->neighbors(v)) {
      if (color_.count(nb)) {
        uint32_t c = color_.at(nb);
        if (c < K) usedColor[c] = true;
      }
    }

    bool assigned = false;
    for (uint32_t c = 0; c < K; ++c) {
      if (!usedColor[c]) {
        color_[v] = c;
        fn_.getVReg(v)->physReg = c;
        assigned = true;
        break;
      }
    }

    if (!assigned) {
      // Actual spill
      int32_t slot = nextSpillSlot_;
      nextSpillSlot_ += 4; // 4 bytes per register
      fn_.getVReg(v)->spillSlot = slot;
      spills.push_back({v, spillCost(v), slot});
    }
  }
}

// ---------------------------------------------------------------------------
// Spill code insertion
// ---------------------------------------------------------------------------

void GraphColoringRA::insertSpillCode(const std::vector<SpillInfo>& spills) {
  if (spills.empty()) return;
  std::unordered_set<ir::VRegID> spillSet;
  std::unordered_map<ir::VRegID, int32_t> slotMap;
  for (auto& si : spills) {
    spillSet.insert(si.vreg);
    slotMap[si.vreg] = si.spillSlot;
  }

  for (auto& bb : fn_.blocks) {
    std::vector<std::unique_ptr<ir::Instruction>> newInstrs;
    for (auto& instr : bb->instrs) {
      // Before instruction: reload each spilled use into a fresh temp vreg
      for (auto& vid : instr->uses) {
        if (spillSet.count(vid)) {
          auto* tempVReg = fn_.newVReg(fn_.getVReg(vid)->type);
          auto reload = std::make_unique<ir::Instruction>(ir::Opcode::LD_LOCAL);
          reload->dest  = tempVReg->id;
          reload->imm   = slotMap.at(vid);
          reload->hasImm = true;
          newInstrs.push_back(std::move(reload));
          vid = tempVReg->id; // Rewrite use to temp
        }
      }

      newInstrs.push_back(std::move(instr));

      // After instruction: spill defined vreg to its slot
      auto& last = newInstrs.back();
      if (last->definesDest() && spillSet.count(last->dest)) {
        auto store = std::make_unique<ir::Instruction>(ir::Opcode::ST_LOCAL);
        store->uses  = {last->dest};
        store->imm   = slotMap.at(last->dest);
        store->hasImm = true;
        newInstrs.push_back(std::move(store));
      }
    }
    bb->instrs = std::move(newInstrs);
  }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

AllocResult GraphColoringRA::run() {
  buildPhase();

  // Main loop: Simplify → Coalesce → Freeze → Spill
  while (true) {
    if (!simplifyWorklist_.empty()) { simplifyPhase(); continue; }
    if (coalescePhase())            { continue; }
    if (freezePhase())              { continue; }
    if (!spillWorklist_.empty())    { spillPhase(); continue; }
    break; // All work-lists empty
  }

  std::vector<SpillInfo> spills;
  selectPhase(spills);

  if (!spills.empty())
    insertSpillCode(spills);

  // Compute result metrics
  uint32_t regsUsed = 0;
  for (auto& [v, c] : color_)
    regsUsed = std::max(regsUsed, c + 1);

  // High-water mark of simultaneously live vregs (from liveness analysis)
  uint32_t maxLive = 0;
  for (auto& bb : fn_.blocks) {
    maxLive = std::max(maxLive,
        static_cast<uint32_t>(la_->blockInfo(bb.get()).liveOut.size()));
  }

  return AllocResult{
    spills.empty(),
    maxLive,
    regsUsed,
    rf_.estimateOccupancy(regsUsed),
    std::move(spills)
  };
}

void AllocResult::dump() const {
  std::cout << "=== Register Allocation Result ===\n"
            << "  Status       : " << (success ? "SUCCESS" : "SPILLED") << "\n"
            << "  Regs used    : " << regsUsed << "\n"
            << "  Max live     : " << maxLiveAtAnyPoint << "\n"
            << "  Occupancy    : " << estimatedOccupancy << " warps/SM\n";
  if (!spills.empty()) {
    std::cout << "  Spills       : " << spills.size() << "\n";
    for (auto& s : spills)
      std::cout << "    vreg %" << s.vreg
                << "  cost=" << s.cost
                << "  slot=" << s.spillSlot << "\n";
  }
}

} // namespace regalloc
} // namespace simt

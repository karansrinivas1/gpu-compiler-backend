# simt-regalloc

**A SIMT-aware graph-coloring register allocator and GPU compiler backend written in C++17.**

This project implements a production-grade compiler backend pipeline targeting NVIDIA GPU architectures — from SSA intermediate representation through Chaitin-Briggs register allocation to PTX assembly emission. It directly addresses one of the hardest and most performance-critical problems in GPU compiler engineering: mapping an unbounded set of virtual registers onto a finite physical register file while maximising hardware occupancy.

---

## Why This Project Exists

Modern GPU kernels are compiled by a backend that must solve a uniquely constrained version of register allocation. Unlike CPU compilers, a GPU register allocator must simultaneously optimise for two competing goals:

1. **Correctness** — every virtual register must be assigned a physical register (or spilled to local memory) such that no two simultaneously-live values share the same register.
2. **Occupancy** — the number of physical registers consumed per thread directly determines how many warps the SM can schedule in parallel. Using fewer registers per thread means more warps, better latency hiding, and higher throughput.

On an NVIDIA Ampere SM (sm_86), the entire SM shares a 256 KB register file (65,536 × 32-bit registers). If each thread uses 64 registers, the SM can run 32 warps simultaneously. If each thread uses 128 registers, occupancy halves to 16 warps. A register allocator that is "merely correct" but spends registers carelessly can cut kernel throughput by 2–4× on memory-bound workloads.

This project was built to explore and implement the full register allocation pipeline with explicit GPU-awareness — something most educational compiler projects never address.

---

## What the Project Does

### Pipeline Overview

```
  Source IR (SSA form)
        │
        ▼
  ┌─────────────────┐
  │  Liveness       │  Backward dataflow: computes LiveIn/LiveOut per block
  │  Analysis       │  and live intervals for every virtual register
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Interference   │  Builds a graph where nodes = vregs, edges = conflict
  │  Graph (RIG)    │  Two vregs interfere ↔ they are simultaneously live
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────┐
  │  Chaitin-Briggs Graph-Coloring Register Allocator   │
  │                                                     │
  │  Phase 1 — Build      liveness + interference graph │
  │  Phase 2 — Simplify   push low-degree nodes         │
  │  Phase 3 — Coalesce   merge copy-related nodes      │
  │  Phase 4 — Freeze     give up coalescing if stuck   │
  │  Phase 5 — Spill      select potential spill node   │
  │  Phase 6 — Select     assign colours; insert spills │
  └────────┬────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────┐
  │  PTX Emitter    │  Translates coloured IR to NVIDIA PTX assembly
  └─────────────────┘
```

---

## Architecture

### Intermediate Representation (`include/simt/IR.h`)

A typed SSA IR designed for GPU compute kernels:

- **Types** — `pred` (1-bit), `s16/s32/s64`, `u16/u32/u64`, `f16/f32/f64`, `b32/b64`
- **Virtual registers** — unbounded SSA register file; each vreg has an assigned type
- **Instructions** — arithmetic, float ops, comparisons, memory (global/shared/local/param), control flow, GPU intrinsics (`threadIdx`, `blockIdx`, `__syncthreads`), PHI nodes
- **CFG** — `BasicBlock` with explicit predecessor/successor lists; `Function` owns the block and vreg lists
- **Module** — top-level container, supports multiple kernel functions

```cpp
// Example: create a typed virtual register
auto* v = fn.newVReg(Type::F32());

// Example: emit an FMA instruction
auto* fma = std::make_unique<Instruction>(Opcode::FFMA);
fma->dest = result->id;
fma->uses = {a->id, b->id, c->id};   // result = a * b + c
block->addInstr(std::move(fma));
```

### Liveness Analysis (`include/simt/Liveness.h`, `lib/Liveness.cpp`)

Classic backward dataflow analysis over the CFG:

```
LiveOut(B)  =  ∪  LiveIn(S)   for every successor S of B
LiveIn(B)   =  Use(B)  ∪  (LiveOut(B) − Def(B))
```

Iterated to a fixed point. PHI-node semantics are handled correctly — a PHI argument is considered live-out of the predecessor block it comes from, not live-in to the PHI's own block. This is the key difference between CPU and GPU SSA liveness that most textbooks underspecify.

Live intervals (contiguous `[start, end)` ranges in a linearised instruction numbering) are also computed and merged, enabling efficient overlap queries used by the interference graph builder.

### Interference Graph (`include/simt/InterferenceGraph.h`, `lib/InterferenceGraph.cpp`)

An adjacency-list graph where:
- Each vertex is a virtual register
- An edge `u — v` means the two vregs are simultaneously live at some program point and therefore cannot share a physical register

Also tracks **affinity edges** (copy-related pairs) — vregs connected by a `COPY` instruction. These are candidates for coalescing: if they don't interfere, merging them eliminates the copy instruction entirely.

The graph supports `coalesce(keep, merge)` which contracts an edge by redirecting all adjacency and affinity relationships from `merge` to `keep` — the core operation in the Briggs coalescing phase.

### Chaitin-Briggs Register Allocator (`include/simt/GraphColoringRA.h`, `lib/GraphColoringRA.cpp`)

Full implementation of the Chaitin-Briggs algorithm, the same foundational algorithm used in production compilers (GCC, LLVM, and NVIDIA's own `ptxas`).

**Simplify** — Nodes with degree < K (number of available physical registers) are provably colourable. They are pushed onto a stack and temporarily removed from the graph, which may reduce the degree of their neighbours and trigger further simplification.

**Coalesce** — Uses the Briggs conservative criterion: two copy-related nodes `u` and `v` are merged only if the combined node has fewer than K neighbours of high degree. This guarantees that coalescing never introduces a spill that wasn't already present.

**Freeze** — When both Simplify and Coalesce are stuck, a low-degree copy-related node is "frozen" (its affinity edges are abandoned) and moved to the Simplify list.

**Spill** — When all remaining nodes have degree ≥ K, a spill candidate is selected using a cost heuristic: `cost = (weighted use count) / degree`. Spilling cheap, heavily-interfering nodes first minimises the number of additional loads/stores inserted.

**Select** — Nodes are popped from the stack and assigned the lowest available colour not used by any neighbour. If no colour is available, the node is actually spilled: spill/reload pseudoinstructions are inserted around each use and definition.

**SIMT occupancy model** — The allocator computes:

```
warps_per_SM = regs_per_SM / (regs_per_thread × warp_size)
```

This gives an estimated hardware occupancy after allocation, letting the compiler report the performance impact of register pressure directly in the build output.

### PTX Emitter (`include/simt/PTXEmitter.h`, `lib/PTXEmitter.cpp`)

Translates the register-allocated IR to NVIDIA PTX assembly:

- Emits `.version`, `.target`, `.address_size` headers
- Declares register banks (`%r<N>`, `%f<N>`, `%rd<N>`, `%p<N>`) sized to the actual physical registers used
- Translates all opcodes to their PTX equivalents with correct type suffixes
- Handles spilled vregs as `[local+offset]` memory references
- Supports GPU intrinsics: `%tid.x/y/z`, `%ctaid.x/y/z`, `%ntid.x`, `bar.sync`

---

## Example Kernels

### Vector Add

```
__global__ void vecAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
```

The driver builds this kernel's IR, runs liveness analysis, constructs the interference graph, allocates registers using Chaitin-Briggs, and emits valid PTX — demonstrating the full pipeline on a real GPU pattern.

### Matrix Multiply Inner Loop

```
__global__ void matMul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.x + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < N; k++)
        sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}
```

This kernel has a loop with a PHI node and significantly higher register pressure — it stresses the allocator's loop-aware spill cost heuristic and coalescing logic.

---

## Building

**Requirements:** C++17 compiler (GCC 9+ or Clang 10+), CMake 3.16+

```bash
git clone https://github.com/karansrinivas1/simt-regalloc.git
cd simt-regalloc

# Configure and build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run the compiler driver (emits PTX for vecAdd and matMul)
./build/gpucc

# Run the test suite
./build/test_regalloc
```

Or compile directly without CMake:

```bash
g++ -std=c++17 -O2 -Iinclude \
    lib/Liveness.cpp lib/InterferenceGraph.cpp \
    lib/GraphColoringRA.cpp lib/PTXEmitter.cpp \
    tools/gpucc.cpp -o gpucc

./gpucc
```

---

## Test Suite

```
=== SIMT Register Allocator Tests ===

[PASS] liveness: %0 live-out of entry
[PASS] liveness: %1 live-out of entry
[PASS] liveness: %2 not live-out of true_bb
[PASS] liveness: %0 not live-out of true_bb
[PASS] interference: %0 ~ %1 (both live at ADD def)
[PASS] coloring K=4: no spills for 4-vreg program
[PASS] coloring K=4: at most 4 physical regs used
[PASS] coloring K=2 with 3 live: spill or success after re-allocation
[PASS] occupancy: 32 regs/thread → 64 warps/SM
[PASS] occupancy: 64 regs/thread → 32 warps/SM
[PASS] occupancy: 128 regs/thread → 16 warps/SM
[PASS] coalescing: no spills
      v0→r0  v1→r0  (coalesced)
[PASS] intervals: [0,10) overlaps [5,15)
[PASS] intervals: [0,10) does not overlap [10,20)
[PASS] intervals: [5,15) overlaps [10,20)

=== Results: 16 passed, 0 failed ===
```

Tests cover: backward dataflow liveness, interference edge insertion, graph coloring without spills, forced spill with K=2, occupancy arithmetic for sm_86, conservative coalescing of copy-related vregs, and live interval overlap detection.

---

## Project Structure

```
simt-regalloc/
├── include/simt/
│   ├── IR.h                  GPU SSA IR — types, vregs, instructions, CFG
│   ├── Liveness.h            Backward dataflow liveness + live intervals
│   ├── InterferenceGraph.h   Register interference and affinity graph
│   ├── GraphColoringRA.h     Chaitin-Briggs allocator with SIMT occupancy model
│   └── PTXEmitter.h          PTX assembly code generator
├── lib/
│   ├── Liveness.cpp
│   ├── InterferenceGraph.cpp
│   ├── GraphColoringRA.cpp
│   └── PTXEmitter.cpp
├── tools/
│   └── gpucc.cpp             Driver: builds vecAdd and matMul IR, runs pipeline
├── test/
│   └── test_regalloc.cpp     16 unit tests (no external test framework)
└── CMakeLists.txt
```

---

## Key Design Decisions

**Why Chaitin-Briggs and not linear scan?** Linear scan (used in some JIT compilers) is faster to run but produces worse allocations under high register pressure. GPU kernels run millions of threads simultaneously — a few extra spills per thread compound into significant memory bandwidth waste. The Chaitin-Briggs algorithm finds globally optimal colorings whenever possible, which is the right tradeoff for an ahead-of-time GPU compiler.

**Why conservative coalescing?** Aggressive coalescing (George's criterion) can in rare cases turn a colourable graph into an uncolourable one by creating new high-degree nodes. Conservative Briggs coalescing guarantees this cannot happen, which is critical for predictable compilation latency in production toolchains.

**Why model occupancy inside the allocator?** Register pressure and occupancy are inseparable on GPUs. An allocator that treats register count as a pure correctness constraint and ignores the throughput impact is incomplete for GPU targets. Making occupancy a first-class output of the allocator lets the compiler pipeline make informed decisions — for example, re-running with a lower register budget to hit a higher occupancy tier.

---

## Relation to Production GPU Compilers

This project implements the same core algorithms found in production GPU compilers:

| Component | This project | Production equivalent |
|---|---|---|
| IR | Typed SSA with PHI nodes | LLVM IR / NVVM IR |
| Liveness | Backward dataflow | LLVM `LiveVariables` / `LiveIntervals` |
| Interference graph | Adjacency list + affinity | LLVM `LiveRegMatrix` |
| Register allocator | Chaitin-Briggs | LLVM `RegAllocGreedy`, `ptxas` |
| Code emission | PTX text | LLVM `AsmPrinter` |
| Occupancy model | regs/SM ÷ (regs/thread × warp) | NVIDIA Occupancy Calculator |

---

## Author

**Karan Srinivas**
M.S. Computer Engineering — Northeastern University
[github.com/karansrinivas1](https://github.com/karansrinivas1) · [linkedin.com/in/karans1](https://linkedin.com/in/karans1/)

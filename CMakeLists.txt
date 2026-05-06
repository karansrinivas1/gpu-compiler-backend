cmake_minimum_required(VERSION 3.16)
project(simt-regalloc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Strict warnings — mirrors LLVM/NVIDIA internal build settings
add_compile_options(
  -Wall -Wextra -Wpedantic
  -Wno-unused-parameter
  $<$<CONFIG:Release>:-O3 -march=native>
  $<$<CONFIG:Debug>:-g -O0 -fsanitize=address,undefined>
)

add_link_options($<$<CONFIG:Debug>:-fsanitize=address,undefined>)

include_directories(include)

# ---------------------------------------------------------------------------
# Backend library
# ---------------------------------------------------------------------------
add_library(simt_backend STATIC
  lib/Liveness.cpp
  lib/InterferenceGraph.cpp
  lib/GraphColoringRA.cpp
  lib/PTXEmitter.cpp
)
target_include_directories(simt_backend PUBLIC include)

# ---------------------------------------------------------------------------
# gpucc driver
# ---------------------------------------------------------------------------
add_executable(gpucc tools/gpucc.cpp)
target_link_libraries(gpucc PRIVATE simt_backend)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_executable(test_regalloc test/test_regalloc.cpp)
target_link_libraries(test_regalloc PRIVATE simt_backend)

add_test(NAME regalloc_suite COMMAND test_regalloc)

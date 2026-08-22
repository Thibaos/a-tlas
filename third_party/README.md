# third-party components

Vendored, distinctly-licensed dependencies. Never relicense or merge their
source into the atlas-rt tree.

## NVIDIA NRD (Real-Time Denoisers) — third_party/nrd

- Pin: tag v4.17.3 (NRD_VERSION 4.17.3, "30 April 2026"), shallow git clone.
- License: proprietary "NVIDIA RTX SDKs LICENSE" (LICENSE.txt in this
  directory's nrd/). NOT MIT. Redistribution terms live in that file;
  attribution notice per its terms: "This software contains source code
  provided by NVIDIA Corporation."
- Build config: static library, SPIR-V-only shaders (DXIL/DXBC off),
  NRD_NORMAL_ENCODING=0 (RGBA8_UNORM world-space normals),
  NRD_ROUGHNESS_ENCODING=1 (linear).
- Built automatically by the crate's build.rs via CMake when
  third_party/nrd/_Bin/Release/NRD.lib is missing. Requires: cmake (MSVC),
  DXC on PATH via VULKAN_SDK (SPIR-V frontend). ShaderMake + MathLib are
  vendored beside it at the exact revisions NRD v4.17.3 pins.

## NVIDIA ShaderMake — third_party/shadermake

- Pin: commit 18f5a344e7ca8fa65daaf079d07bc8ce38453e05 (the commit NRD
  v4.17.3 fetches). License: MIT (in repo).

## NVIDIA MathLib — third_party/mathlib

- Pin: tag v11. License: MIT (in repo).

## Updating

Bump the pin, re-probe struct sizes (third_party/probe_sizes.cpp pattern)
against the new headers, and update the size assertions in
src/render/nrd/sys.rs plus the front-end mirror in shaders/common/nrd.glsl.

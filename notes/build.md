# Local build and module install (verified 2026-11-25)

Host: Strix Halo (Ryzen AI MAX+ 395, gfx1151), 128 GiB unified memory.

Gufo base: `v0.36.0` (`4a7d8c6`), extension API **0.3.42**.

## Toolchain

Builds run in the Nix devshell of the pinned Gufo checkout, inside a
`nixos/nix` container (the same shape the in-tree work on this box uses):

```sh
podman run --rm --security-opt label=disable \
  -e NIX_CONFIG=$'experimental-features = nix-command flakes\nsandbox = false' \
  -v gufo-nix:/nix -v ~/ling32-work:/work -v ~/Models:/models:ro \
  -w /work/gufo-src docker.io/nixos/nix:latest \
  sh -c 'nix develop --accept-flake-config -c <command>'
```

The devshell provides CMake 4.4.4, Ninja 1.13.2, hipcc (ROCm 7.16, clang 24,
TheRock 10.1) and targets `gfx1151`.

## Configure and build

```sh
cmake -S /work/gufo-src -B /work/ext-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF -DENGINE_ENABLE_HIP=ON \
  -DGUFO_EXTENSION_DIRS=/work/ext-build -DGUFO_EXTENSIONS=ling3_air
cmake --build /work/ext-hip --target gufo --parallel 20
```

Result: `/work/ext-hip/gufo` (extension compiled in) and
`/work/ext-hip/sdk` (the extension SDK from the same build).

## Module install into a fresh prefix

```sh
export GUFO_SDK_DIR=/work/ext-hip/sdk GUFO_EXTENSIONS_DIR=/work/ext-store
/work/ext-hip/gufo extension install /work/ext-build --yes --replace-builtin
/work/ext-hip/gufo extension list            # ling3_air 0.1.0 [installed, ready]
/work/ext-hip/gufo diagnose --section extensions --json
```

`--replace-builtin` is needed only because this development binary already has
ling3_air compiled in; a stock Gufo 0.36.0 takes the module with a plain
`install`. Both were checked: the module is built from the extension source by
Gufo itself (CMake configure + Ninja in `.staging`), and the installed module is
what `diagnose` then reports (`source: module:.../gufo-ext-ling3_air.so`).

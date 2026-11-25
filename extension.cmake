# Ling 3.2 Air model extension.

# Host-only: manifest-driven metadata, probes and GGUF configuration parsing.
# No HIP, so diagnostics and probes work on machines without a GPU.
gufo_extension_target(
  NAME gufo_ling3_air
  SOURCES
    src/extension.cpp
    src/probe.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
)

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/ling3_air)

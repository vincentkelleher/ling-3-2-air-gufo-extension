# Ling 3.2 Air model extension.

# Host-only: manifest-driven metadata, probes, GGUF configuration parsing,
# tokenizer and chat framing. No HIP, so diagnostics and probes work on
# machines without a GPU.
gufo_extension_target(
  NAME gufo_ling3_air
  SOURCES
    src/extension.cpp
    src/probe.cpp
    src/text/unicode.cpp
    src/text/unicode_tables.cpp
    src/text/pretokenizer.cpp
    src/text/tokenizer.cpp
    src/text/chat_template.cpp
    src/config.cpp
    src/weights.cpp
    src/preprocess.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
)
if(TARGET gufo_ling3_air)
  # Sources include each other as "src/...", as Gufo's own extensions do.
  target_include_directories(gufo_ling3_air PUBLIC ${GUFO_EXTENSION_BUILD_ROOT})
  # NFC normalization matches the reference tokenizer's normalizer. ICU is
  # what Gufo itself links; when the build does not provide it the tokenizer
  # runs unnormalized and reports that (see text::HasIcuNormalization).
  include(CheckIncludeFileCXX)
  find_package(ICU COMPONENTS uc QUIET)
  if(ICU_FOUND)
    target_compile_definitions(gufo_ling3_air PRIVATE LING3_HAVE_ICU=1)
    target_link_libraries(gufo_ling3_air PRIVATE ICU::uc)
  endif()
endif()

# The runner: engine orchestration plus the HIP kernels.
gufo_extension_target(
  NAME gufo_ling3_air_hip
  SOURCES
    src/engine.cpp
    src/drivers.cpp
    src/vision.cpp
    src/runner.cpp
  LINK_LIBRARIES
    gufo_ling3_air
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)
if(TARGET gufo_ling3_air_hip)
  target_include_directories(gufo_ling3_air_hip PUBLIC ${GUFO_EXTENSION_BUILD_ROOT})
  gufo_extension_hip_sources(gufo_ling3_air_hip SOURCES
    src/hip/kernels.hip
    src/hip/layers.hip
    src/hip/attention.hip
    src/hip/moe.hip
    src/hip/ops.hip
    src/hip/vision.hip)
endif()

# A replacement module can be loaded into a Gufo binary with an older Ling
# built in. On ELF, hide our archive definitions from dynamic lookup in the
# module: otherwise its new methods can receive an old Engine::Impl allocated
# by the executable's Engine::Create. Leave the SDK entry points (generated
# outside these archives) and imports from the host alone.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  foreach(lib IN ITEMS gufo_ling3_air gufo_ling3_air_hip)
    if(TARGET ${lib})
      target_link_options(${lib} INTERFACE
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:LINKER:--exclude-libs,lib${lib}.a>")
    endif()
  endforeach()
endif()

# End-to-end forward: load, decode, report tokens and rate.
gufo_extension_target(
  NAME ling3_air_forward_test
  SOURCES
    tests/forward_test.cpp
  LINK_LIBRARIES
    gufo_ling3_air_hip
    gufo_ling3_air
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)
if(TARGET ling3_air_forward_test)
  add_test(NAME ling3_air.forward COMMAND ling3_air_forward_test)
  set_tests_properties(ling3_air.forward PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()

# Dequantization agreement with Gufo's reference on real weight rows.
gufo_extension_target(
  NAME ling3_air_dot_test
  SOURCES
    tests/dot_vs_gufo_test.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
)
if(TARGET ling3_air_dot_test)
  add_test(NAME ling3_air.dot COMMAND ling3_air_dot_test)
  set_tests_properties(ling3_air.dot PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Probe behaviour: claims the pinned file, refuses everything else.
gufo_extension_target(
  NAME ling3_air_probe_test
  SOURCES
    tests/probe_test.cpp
  LINK_LIBRARIES
    gufo_ling3_air
    gufo_ext_runtime
    gufo_ext_core
)
if(TARGET ling3_air_probe_test)
  add_test(NAME ling3_air.probe COMMAND ling3_air_probe_test)
  set_tests_properties(ling3_air.probe PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Chat framing against fixtures rendered by the model's own jinja template.
gufo_extension_target(
  NAME ling3_air_chat_test
  SOURCES
    tests/chat_template_test.cpp
  LINK_LIBRARIES
    gufo_ling3_air
    gufo_ext_runtime
    gufo_ext_core
)
if(TARGET ling3_air_chat_test)
  target_compile_definitions(ling3_air_chat_test PRIVATE
    LING3_FIXTURES_DIR="${GUFO_EXTENSION_BUILD_ROOT}/fixtures")
  add_test(NAME ling3_air.chat COMMAND ling3_air_chat_test)
endif()

# Vision encoder against the numpy reference from tools/make_vision_fixture.py.
gufo_extension_target(
  NAME ling3_air_vision_test
  SOURCES
    tests/vision_test.cpp
  LINK_LIBRARIES
    gufo_ling3_air_hip
    gufo_ling3_air
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)
if(TARGET ling3_air_vision_test)
  target_compile_definitions(ling3_air_vision_test PRIVATE
    LING3_FIXTURES_DIR="${GUFO_EXTENSION_BUILD_ROOT}/fixtures")
  add_test(NAME ling3_air.vision COMMAND ling3_air_vision_test)
  set_tests_properties(ling3_air.vision PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
endif()

# Image preprocessing against numpy and PIL references.
gufo_extension_target(
  NAME ling3_air_preprocess_test
  SOURCES
    tests/preprocess_test.cpp
  LINK_LIBRARIES
    gufo_ling3_air
    gufo_ext_runtime
    gufo_ext_core
)
if(TARGET ling3_air_preprocess_test)
  add_test(NAME ling3_air.preprocess COMMAND ling3_air_preprocess_test)
  set_tests_properties(ling3_air.preprocess PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
endif()

# Image conditioning end to end (preprocess -> tower -> trunk).
gufo_extension_target(
  NAME ling3_air_image_forward_test
  SOURCES
    tests/image_forward_test.cpp
  LINK_LIBRARIES
    gufo_ling3_air_hip
    gufo_ling3_air
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)
if(TARGET ling3_air_image_forward_test)
  add_test(NAME ling3_air.image_forward COMMAND ling3_air_image_forward_test)
  set_tests_properties(ling3_air.image_forward PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()

# MTP speculative decoding must reproduce plain greedy output exactly.
gufo_extension_target(
  NAME ling3_air_mtp_test
  SOURCES
    tests/mtp_test.cpp
  LINK_LIBRARIES
    gufo_ling3_air_hip
    gufo_ling3_air
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)
if(TARGET ling3_air_mtp_test)
  add_test(NAME ling3_air.mtp COMMAND ling3_air_mtp_test)
  # Asserts what is proven (state restore is bit-exact, logits identical); the speculative
  # stream's divergence is printed rather than asserted, because the loop is unproven.
  set_tests_properties(ling3_air.mtp PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
endif()

# Config and weight-map validation against a real GGUF.
gufo_extension_target(
  NAME ling3_air_config_test
  SOURCES
    tests/config_test.cpp
  LINK_LIBRARIES
    gufo_ling3_air
    gufo_ext_runtime
    gufo_ext_core
)
if(TARGET ling3_air_config_test)
  add_test(NAME ling3_air.config COMMAND ling3_air_config_test)
  set_tests_properties(ling3_air.config PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Tokenizer conformance against the reference tokenizer.json fixture.
gufo_extension_target(
  NAME ling3_air_tokenizer_test
  SOURCES
    tests/tokenizer_test.cpp
  LINK_LIBRARIES
    gufo_ling3_air
    gufo_ext_runtime
    gufo_ext_core
)
if(TARGET ling3_air_tokenizer_test)
  target_compile_definitions(ling3_air_tokenizer_test PRIVATE
    LING3_FIXTURES_DIR="${GUFO_EXTENSION_BUILD_ROOT}/fixtures")
  add_test(NAME ling3_air.tokenizer COMMAND ling3_air_tokenizer_test)
  set_tests_properties(ling3_air.tokenizer PROPERTIES SKIP_RETURN_CODE 77)
endif()

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/ling3_air)

#include <gufo/ext/generated/ling3_air_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/probe.hpp"

// The provider is registered in two pieces, as Gufo's own text extensions do:
//
//   * AddProvider() publishes the metadata from extension.json. It needs no
//     GPU, so `gufo diagnose --section extensions` describes the provider on a
//     CPU-only machine.
//   * AddTextProvider() publishes the probe and, once the engine exists, its
//     runner and drivers. It is only compiled when the host built HIP: a text
//     runner cannot serve without it.
//
// Until the engine lands, this extension announces itself and claims no
// weights, so a Ling file is refused rather than half-served.

void ling3_air_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_ling3_air::kProvider_ling3_air);
}

void ling3_air_register(gufo::ext::Registry&) {}

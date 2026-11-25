# Third-party notices

## Gufo

This extension is built against and loaded by
[Gufo](https://github.com/gufo-org/gufo), MIT licensed. Gufo publishes the
extensions API, the GGUF reader, sampling and the HIP helpers this extension
uses; those remain Gufo's code and copyright.

## Model weights

The weights this extension serves are not part of this repository and are not
redistributed by it.

- Ling 3.2 Air, `inclusionAI/Ling-3.2-air` on Hugging Face. The license terms
  are the publisher's; see that model card before downloading.
- The GGUF quantization and the multimodal projector used here are
  `bartowski/Ling-3.2-air-GGUF` (revision
  `2073edb5da4296cb37b334584b858aa6141e352f`), produced with llama.cpp.

The pinned file list, sizes and SHA-256 digests live in `models.lock.json`,
which `gufo model install` checks before serving.

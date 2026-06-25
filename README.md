# graft

[![CI](https://github.com/nathanm82/graft/actions/workflows/ci.yml/badge.svg)](https://github.com/nathanm82/graft/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Lightweight visual connectors that bridge pretrained vision encoders to LLMs
for visual instruction tuning.

A *connector* (a.k.a. projector / adapter / resampler) is the small trainable
module that maps frozen vision-encoder features into a language model's token
embedding space. `graft` collects the common architectures behind one tiny,
typed API so you can swap them with a single string.

## Install

```bash
pip install graft        # pulls in torch
```

## Quickstart

```python
import torch
from graft import build_connector

# patch features from a frozen CLIP ViT-L/14 (B, N, D)
features = torch.randn(2, 256, 1024)

# project them into a 4096-d Llama embedding space
connector = build_connector("mlp", input_dim=1024, output_dim=4096)
tokens = connector(features)          # (2, 256, 4096)
```

Don't want to look up hidden sizes? Build from model names instead:

```python
connector = build_connector("perceiver", encoder="clip-vit-l-14",
                            llm="llama-2-7b", num_query_tokens=32)
```

## Connectors

| name            | what it does                                   | output tokens |
| --------------- | ---------------------------------------------- | ------------- |
| `linear`        | single linear projection (LLaVA v1)            | unchanged     |
| `mlp`           | 2-layer GELU projector (LLaVA v1.5)            | unchanged     |
| `avgpool`       | adaptive average pool + projection             | reduced       |
| `pixel_shuffle` | space-to-depth downsample + projection         | `/ s²`        |
| `perceiver`     | learned queries cross-attend (Flamingo)        | fixed         |
| `qformer`       | query transformer, self + cross attention      | fixed         |

## Why

The connector is the smallest, most-swapped part of a VLM, but it usually
lives buried inside a modeling file. Pulling it out makes the choice explicit
and makes the visual **token budget** — how much LLM context each image costs —
a first-class number you can compare. See [docs/design-notes.md](docs/design-notes.md).

## Docs

- [Architecture](docs/architecture.md)
- [Usage](docs/usage.md)
- [API reference](docs/api-reference.md)
- [Design notes](docs/design-notes.md)
- [Runnable examples](examples/)

## License

MIT

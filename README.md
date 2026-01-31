# graft

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

## Status

Pre-1.0 and moving quickly, but the connector contract below is stable.

## License

MIT

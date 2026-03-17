# Usage

## Building a connector

Three equivalent ways to build the same module:

```python
from graft import ConnectorConfig, build_connector

# 1. name + keyword dims
build_connector("mlp", input_dim=1024, output_dim=4096)

# 2. a plain dict (e.g. parsed from JSON/YAML)
build_connector({"name": "mlp", "input_dim": 1024, "output_dim": 4096})

# 3. an explicit config object
build_connector(ConnectorConfig(name="mlp", input_dim=1024, output_dim=4096))
```

Or resolve the dimensions from model names:

```python
build_connector("perceiver", encoder="clip-vit-l-14", llm="llama-2-7b",
                num_query_tokens=32)
```

## In a training loop

A connector is just an `nn.Module`. Freeze the encoder and the LLM, train the
connector:

```python
connector = build_connector("mlp", encoder="clip-vit-l-14", llm="llama-2-7b")
optimizer = torch.optim.AdamW(connector.parameters(), lr=1e-3)

image_features = vision_encoder(pixel_values)      # frozen, (B, N, 1024)
visual_tokens = connector(image_features)          # (B, M, 4096)
inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
loss = llm(inputs_embeds=inputs_embeds, labels=labels).loss
```

`connector.freeze()` / `connector.unfreeze()` toggle `requires_grad` if you
want to stage training.

## Inspecting

```python
from graft import connector_summary

print(connector_summary(connector, num_input_tokens=256))
```

```
connector     mlp
input_dim     1024
output_dim    4096
parameters    20,979,712
trainable     20,979,712
input_tokens  256
output_tokens 256
compression   1.00x
```

## Saving and loading

```python
from graft import save_connector, load_connector

save_connector(connector, "projector.pt")
connector = load_connector("projector.pt")   # rebuilds from the stored config
```

## CLI

```bash
graft list
graft specs
graft describe perceiver
graft plan --connector perceiver --encoder clip-vit-l-14 --llm llama-2-7b --num-query-tokens 32
```

# Design notes

## Why a separate library?

The connector is the smallest, most-swapped part of a vision-language model,
yet in most code-bases it is a few lines buried inside a `modeling_*.py` file.
Pulling it out makes the choice explicit and lets you A/B different
architectures (linear vs. MLP vs. resampler) by changing one string.

## Token budget is the real knob

For instruction tuning the connector's most important property is often *how
many tokens it emits*, because every visual token competes with text tokens
for the LLM's context window. That is why `num_output_tokens` is part of the
base contract and why `connector_summary` reports a compression ratio:

| connector       | 576 input tokens -> output |
| --------------- | -------------------------- |
| `linear`/`mlp`  | 576 (1.0x)                 |
| `pixel_shuffle` (x2) | 144 (4.0x)            |
| `perceiver`/`qformer` | `num_query_tokens` (fixed) |

## Keeping it torch-only

The core deliberately depends on nothing but `torch`. `pyyaml` is an optional
extra used only by `ConnectorConfig.from_yaml`. There is no trainer, no
dataloader, and no model zoo — those belong to the framework that *uses* a
connector, not to the connector itself.

## Known limitations

- The Q-Former here is a trimmed-down variant: it has no text-conditioned
  cross-attention, just learned queries over image features.
- `pixel_shuffle` assumes a square token grid; encoders with a CLS token
  should strip it first.
- Weight init follows the common truncated-normal recipe; it is not tuned
  per-architecture.

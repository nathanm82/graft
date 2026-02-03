# Examples

Each script is self-contained and runs on CPU in a second or two.

```bash
python examples/01_mlp_projector.py
python examples/02_perceiver_resampler.py
python examples/03_build_from_config.py
python examples/04_token_budget.py
```

- **01_mlp_projector** — the classic LLaVA-style MLP projector.
- **02_perceiver_resampler** — compress hundreds of tokens to a fixed budget.
- **03_build_from_config** — build a connector from a plain dict.
- **04_token_budget** — compare how much LLM context each connector spends.

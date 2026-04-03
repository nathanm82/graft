# Security Policy

## Supported versions

graft is pre-1.0; only the latest released version receives fixes.

## Reporting a vulnerability

Please report security issues privately using GitHub's
["Report a vulnerability"](https://github.com/nathanm82/graft/security/advisories/new)
flow rather than opening a public issue.

A couple of notes specific to this library:

- `load_connector` uses `torch.load(..., weights_only=True)`, so loading a
  checkpoint will not execute arbitrary pickled code. Still, only load
  checkpoints you trust.
- Connectors are plain `nn.Module`s and do not execute any code from a config.

I aim to acknowledge reports within a few days.

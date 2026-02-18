"""Build a connector from a plain dict (e.g. loaded from JSON or YAML)."""

import torch

from graft import build_connector

CONFIG = {
    "name": "pixel_shuffle",
    "input_dim": 1024,
    "output_dim": 4096,
    "scale_factor": 2,
}


def main() -> None:
    connector = build_connector(CONFIG)

    # 576 tokens == a 24x24 grid; pixel-shuffle x2 -> 12x12 == 144 tokens.
    features = torch.randn(1, 576, 1024)
    print("output:", tuple(connector(features).shape))


if __name__ == "__main__":
    main()

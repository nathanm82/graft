"""Project CLIP features into a Llama embedding space with an MLP connector."""

import torch

from graft import build_connector


def main() -> None:
    # Pretend these are patch features from a frozen CLIP ViT-L/14.
    features = torch.randn(2, 256, 1024)

    connector = build_connector("mlp", input_dim=1024, output_dim=4096)
    tokens = connector(features)

    print("input :", tuple(features.shape))
    print("output:", tuple(tokens.shape))
    print("params:", connector.num_parameters())


if __name__ == "__main__":
    main()

"""Compress 729 SigLIP tokens down to 32 with a Perceiver resampler."""

import torch

from graft import ConnectorConfig, build_connector, connector_summary


def main() -> None:
    features = torch.randn(1, 729, 1152)

    cfg = ConnectorConfig(
        name="perceiver",
        input_dim=1152,
        output_dim=4096,
        num_query_tokens=32,
        num_heads=8,
        depth=2,
    )
    connector = build_connector(cfg)

    print(connector_summary(connector, num_input_tokens=729))
    print("output:", tuple(connector(features).shape))


if __name__ == "__main__":
    main()

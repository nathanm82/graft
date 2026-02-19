"""Compare how much LLM context each connector spends on a single image."""

from graft import ConnectorConfig, build_connector, estimate_sequence_length

# CLIP ViT-L/14 at 336px produces a 24x24 grid of patch tokens.
ENCODER_TOKENS = 576


def main() -> None:
    options = {
        "mlp (keep all)": ConnectorConfig(name="mlp", input_dim=1024, output_dim=4096),
        "pixel_shuffle x2": ConnectorConfig(
            name="pixel_shuffle", input_dim=1024, output_dim=4096, extra={"scale_factor": 2}
        ),
        "perceiver-64": ConnectorConfig(
            name="perceiver", input_dim=1024, output_dim=4096, num_query_tokens=64, num_heads=8
        ),
    }

    for label, cfg in options.items():
        connector = build_connector(cfg)
        visual = connector.num_output_tokens(ENCODER_TOKENS)
        total = estimate_sequence_length(connector, ENCODER_TOKENS, num_text_tokens=128)
        print(f"{label:18s} -> {visual:4d} visual tokens, {total} total (with 128 text tokens)")


if __name__ == "__main__":
    main()

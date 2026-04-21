from graft import (
    ConnectorConfig,
    build_connector,
    connector_summary,
    estimate_sequence_length,
)


def test_summary_param_counts_and_ratio():
    conn = build_connector("linear", input_dim=16, output_dim=32)
    summary = connector_summary(conn, num_input_tokens=10)
    assert summary.num_parameters == 16 * 32 + 32
    assert summary.num_output_tokens == 10
    assert summary.compression_ratio == 1.0
    assert "connector" in str(summary)


def test_summary_compression_for_resampler():
    cfg = ConnectorConfig(
        name="perceiver", input_dim=16, output_dim=32, num_query_tokens=8, num_heads=4
    )
    summary = connector_summary(build_connector(cfg), num_input_tokens=64)
    assert summary.num_output_tokens == 8
    assert summary.compression_ratio == 8.0


def test_summary_without_input_tokens():
    summary = connector_summary(build_connector("linear", input_dim=8, output_dim=16))
    assert summary.num_output_tokens is None
    assert summary.compression_ratio is None
    assert summary.num_parameters > 0


def test_estimate_sequence_length():
    cfg = ConnectorConfig(
        name="perceiver", input_dim=16, output_dim=32, num_query_tokens=8, num_heads=4
    )
    conn = build_connector(cfg)
    assert estimate_sequence_length(conn, num_input_tokens=64, num_text_tokens=10) == 18

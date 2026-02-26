"""Human-readable summaries of a connector."""

from __future__ import annotations

from dataclasses import dataclass

from graft.base import Connector


@dataclass
class ConnectorSummary:
    """A small report describing a built connector."""

    name: str
    input_dim: int
    output_dim: int
    num_parameters: int
    trainable_parameters: int
    num_input_tokens: int | None
    num_output_tokens: int | None
    compression_ratio: float | None

    def as_table(self) -> str:
        rows: list[tuple[str, object]] = [
            ("connector", self.name),
            ("input_dim", self.input_dim),
            ("output_dim", self.output_dim),
            ("parameters", f"{self.num_parameters:,}"),
            ("trainable", f"{self.trainable_parameters:,}"),
            ("input_tokens", self.num_input_tokens),
            ("output_tokens", self.num_output_tokens),
            ("compression", f"{self.compression_ratio:.2f}x" if self.compression_ratio else "-"),
        ]
        width = max(len(key) for key, _ in rows)
        return "\n".join(f"{key.ljust(width)}  {value}" for key, value in rows)

    def __str__(self) -> str:
        return self.as_table()


def connector_summary(
    connector: Connector, num_input_tokens: int | None = None
) -> ConnectorSummary:
    """Build a :class:`ConnectorSummary` for ``connector``.

    If ``num_input_tokens`` is given (or present on the config), the output
    token count and compression ratio are computed too.
    """
    config = connector.config
    n_in = num_input_tokens if num_input_tokens is not None else config.num_input_tokens
    n_out: int | None = None
    ratio: float | None = None
    if n_in is not None:
        try:
            n_out = connector.num_output_tokens(n_in)
        except Exception:
            n_out = None
        if n_out:
            ratio = n_in / n_out
    return ConnectorSummary(
        name=config.name,
        input_dim=connector.input_dim,
        output_dim=connector.output_dim,
        num_parameters=connector.num_parameters(),
        trainable_parameters=connector.num_parameters(trainable_only=True),
        num_input_tokens=n_in,
        num_output_tokens=n_out,
        compression_ratio=ratio,
    )


def estimate_sequence_length(
    connector: Connector, num_input_tokens: int, num_text_tokens: int = 0
) -> int:
    """Total LLM sequence length once the visual tokens are inserted."""
    return connector.num_output_tokens(num_input_tokens) + num_text_tokens

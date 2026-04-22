"""Tiny CPU forward-pass benchmark for each connector.

Run with: python benchmarks/bench_connectors.py
"""

import time

import torch

from graft import ConnectorConfig, build_connector

DIM = 1024
OUT_DIM = 4096
TOKENS = 576  # 24x24 grid


def bench(name: str, cfg: ConnectorConfig, runs: int = 20) -> None:
    conn = build_connector(cfg).eval()
    x = torch.randn(1, TOKENS, DIM)

    with torch.no_grad():
        conn(x)  # warmup
        start = time.perf_counter()
        for _ in range(runs):
            out = conn(x)
        elapsed_ms = (time.perf_counter() - start) / runs * 1e3

    # print(conn)  # debug
    print(f"{name:14s} {conn.num_parameters():>12,} params  {elapsed_ms:6.2f} ms/iter  -> {tuple(out.shape)}")


def main() -> None:
    configs = {
        "linear": ConnectorConfig(name="linear", input_dim=DIM, output_dim=OUT_DIM),
        "mlp": ConnectorConfig(name="mlp", input_dim=DIM, output_dim=OUT_DIM),
        "avgpool": ConnectorConfig(name="avgpool", input_dim=DIM, output_dim=OUT_DIM, num_query_tokens=64),
        "pixel_shuffle": ConnectorConfig(name="pixel_shuffle", input_dim=DIM, output_dim=OUT_DIM, extra={"scale_factor": 2}),
        "perceiver": ConnectorConfig(name="perceiver", input_dim=DIM, output_dim=OUT_DIM, num_query_tokens=64, num_heads=8),
        "qformer": ConnectorConfig(name="qformer", input_dim=DIM, output_dim=OUT_DIM, num_query_tokens=64, num_heads=8),
    }
    for name, cfg in configs.items():
        bench(name, cfg)


if __name__ == "__main__":
    main()

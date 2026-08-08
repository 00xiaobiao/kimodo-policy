#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="LeRobot dataset root path")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    parquet_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    vals = []
    total = 0

    for p in parquet_files:
        table = pq.read_table(p, columns=["action"])
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)

        if action.ndim != 2 or action.shape[1] <= 39:
            raise ValueError(f"Bad action shape in {p}: {action.shape}")

        hand = action[:, 38:40]
        vals.append(hand)
        total += hand.shape[0]

    hand = np.concatenate(vals, axis=0)

    print(f"dataset: {root}")
    print(f"frames: {total}")
    print("dim38 unique:", np.unique(hand[:, 0]).tolist())
    print("dim39 unique:", np.unique(hand[:, 1]).tolist())

    for i, name in [(0, "dim38"), (1, "dim39")]:
        x = hand[:, i]
        print(f"\n{name}:")
        print("  min:", float(x.min()))
        print("  max:", float(x.max()))
        print("  mean:", float(x.mean()))
        print("  std:", float(x.std()))
        print("  count_0:", int((x == 0).sum()))
        print("  count_1:", int((x == 1).sum()))
        print("  non_binary:", int(((x != 0) & (x != 1)).sum()))
        print("  q01/q10/q50/q90/q99:", np.quantile(x, [0.01, 0.10, 0.50, 0.90,
0.99]).tolist())


if __name__ == "__main__":
    main()

"""Minimal RELION STAR file reader/writer (3.0 and 3.1+ formats)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def read_star(path: str) -> dict[str, pd.DataFrame]:
    """Return {block_name: DataFrame} for every loop_ block in the file."""
    blocks: dict[str, pd.DataFrame] = {}
    name, cols, rows, in_loop = None, [], [], False

    def flush():
        if name is not None and cols:
            df = pd.DataFrame(rows, columns=cols)
            for c in df.columns:
                conv = pd.to_numeric(df[c], errors="coerce")
                if conv.notna().all():
                    df[c] = conv
            blocks[name] = df

    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("data_"):
                flush()
                name, cols, rows, in_loop = line[5:], [], [], False
            elif line.startswith("loop_"):
                in_loop = True
            elif line.startswith("_"):
                cols.append(line.split()[0][1:])
            elif in_loop:
                rows.append(line.split())
    flush()
    return blocks


def write_star(path: str, blocks: dict[str, pd.DataFrame]) -> None:
    with open(path, "w") as fh:
        fh.write("\n# version 30001\n")
        for name, df in blocks.items():
            fh.write(f"\ndata_{name}\n\nloop_ \n")
            for i, c in enumerate(df.columns):
                fh.write(f"_{c} #{i + 1} \n")
            out = df.copy()
            for c in out.columns:
                if out[c].dtype == object and out[c].isna().any():     # numbers with gaps (None): an empty field
                    num = pd.to_numeric(out[c], errors="coerce")       # would shift every later column
                    if num.notna().sum() == out[c].notna().sum():
                        out[c] = num.astype(float)
                    else:
                        out[c] = out[c].fillna("None")
                if pd.api.types.is_float_dtype(out[c]):
                    out[c] = out[c].map("{:.6f}".format) if len(out) < 1000 else np.char.mod("%.6f", out[c].to_numpy())
            out.to_csv(fh, sep=" ", header=False, index=False)
            fh.write(" \n")

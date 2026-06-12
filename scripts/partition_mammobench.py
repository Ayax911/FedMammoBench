"""Partition the Mammo-Bench CSV into per-node manifests + server test set.

Strategy: source_dataset as the natural partitioning axis (Non-IID).
Each node represents a different geographic/institutional source.

  Node 0  (cmmd)             — China,           ~5 202 rows
  Node 1  (dmid)             — Unknown,          ~  757 rows
  Node 2  (ibia)             — Unknown origin,  ~3 577 rows
  Node 3  (cdd-cesm,kau-bcmd)— Egypt + S.Arabia,~3 137 rows
  Node 4  (ddsm)             — USA, CBIS-DDSM, ~10 400 rows
  Server  (inbreast)         — Portugal,          ~  410 rows
                               (institution NOT seen by any client)

Flags
-----
  --nodes 2   → uses 2-node layout (node0=cmmd, node1=dmid, server_test=rest)
  --nodes 5   → uses 5-node layout (default, described above)

Usage
-----
  python scripts/partition_mammobench.py \\
      --csv data/mammobench/mammo-bench.csv \\
      --out manifests/

  python scripts/partition_mammobench.py \\
      --csv data/mammobench/mammo-bench.csv \\
      --out manifests/ \\
      --nodes 2

Output files
------------
  manifests/node0_manifest.csv  …  node<N>_manifest.csv
  manifests/server_test_manifest.csv
  manifests/partition_summary.txt
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Partition maps
# ---------------------------------------------------------------------------
PARTITION_MAP_2NODES: dict[str, list[str]] = {
    "node0": ["cmmd"],                                              # China
    "node1": ["dmid"],                                             # Desconocido
    "server_test": ["ibia", "kau-bcmd", "cdd-cesm", "inbreast", "ddsm"],
}

PARTITION_MAP_5NODES: dict[str, list[str]] = {
    "node0": ["cmmd"],                                              # China
    "node1": ["dmid"],                                             # Desconocido
    "node2": ["ibia"],                                             # Origen desconocido
    "node3": ["cdd-cesm", "kau-bcmd"],                            # Egipto + Arabia Saudita
    "node4": ["ddsm"],                                             # USA / CBIS-DDSM
    "server_test": ["inbreast"],                                   # Portugal — eval servidor
}

SUSPICIOUS_LABEL = "Suspicious Malignant"


def _stats(df: pd.DataFrame) -> str:
    mal = (df["classification"] == "Malignant").sum()
    ben = (df["classification"] == "Benign").sum()
    nor = (df["classification"] == "Normal").sum()
    total = mal + ben + nor
    pct = mal / total * 100 if total else 0.0
    patients = df["source_subjectID"].nunique()
    return (
        f"rows={len(df)}  patients={patients}  "
        f"Malignant={mal}  Benign={ben}  Normal={nor}  "
        f"malign_rate={pct:.1f}%"
    )


def partition(csv_path: Path, out_dir: Path, *, num_nodes: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if num_nodes == 2:
        partition_map = PARTITION_MAP_2NODES
    elif num_nodes == 5:
        partition_map = PARTITION_MAP_5NODES
    else:
        raise ValueError(f"--nodes must be 2 or 5, got {num_nodes}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}  (layout: {num_nodes} nodes)")

    # Drop ambiguous label rows (235 cases in the dataset).
    n_before = len(df)
    df = df[df["classification"] != SUSPICIOUS_LABEL].copy()
    dropped = n_before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with label='{SUSPICIOUS_LABEL}'")

    sources_in_csv = set(df["source_dataset"].unique())
    all_mapped_sources: set[str] = set()
    for sources in partition_map.values():
        all_mapped_sources.update(sources)

    unmapped = sources_in_csv - all_mapped_sources
    if unmapped:
        print(f"WARNING: sources not in partition map (will be ignored): {unmapped}")

    summary_lines: list[str] = [
        "Mammo-Bench partition summary",
        "=" * 60,
        f"Input CSV  : {csv_path}",
        f"Num nodes  : {num_nodes}",
        f"Total rows (after dropping Suspicious Malignant): {len(df)}",
        "",
    ]

    for partition_name, sources in partition_map.items():
        subset = df[df["source_dataset"].isin(sources)].copy()

        # Determine output filename.
        if partition_name == "server_test":
            fname = "server_test_manifest.csv"
        else:
            fname = f"{partition_name}_manifest.csv"

        out_path = out_dir / fname
        subset.to_csv(out_path, index=False)

        stats = _stats(subset)
        line = f"{partition_name:15s} ({', '.join(sources)}): {stats}"
        summary_lines.append(line)
        print(f"Saved {len(subset):6d} rows -> {out_path}")

    # Verify no row was assigned to more than one partition.
    assigned_idxs: list[int] = []
    for sources in partition_map.values():
        assigned_idxs.extend(df[df["source_dataset"].isin(sources)].index.tolist())
    if len(assigned_idxs) != len(set(assigned_idxs)):
        print("ERROR: some rows appear in more than one partition!")
    else:
        summary_lines.append("")
        summary_lines.append(
            f"Total rows across all partitions: {len(assigned_idxs)} "
            f"(matches input: {len(assigned_idxs) == len(df)})"
        )

    summary_path = out_dir / "partition_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\nSummary written to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partition Mammo-Bench CSV into per-node manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              # 5 nodos (por defecto — para exp con 5 hospitales):
              python scripts/partition_mammobench.py \\
                  --csv data/mammobench/mammo-bench.csv \\
                  --out manifests/

              # 2 nodos (experimento de referencia exp01):
              python scripts/partition_mammobench.py \\
                  --csv data/mammobench/mammo-bench.csv \\
                  --out manifests/ \\
                  --nodes 2
            """
        ),
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Path to mammo-bench.csv",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for partition CSVs",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=5,
        choices=[2, 5],
        help="Number of client nodes (2 or 5, default 5).",
    )
    args = parser.parse_args()
    partition(args.csv, args.out, num_nodes=args.nodes)


if __name__ == "__main__":
    main()

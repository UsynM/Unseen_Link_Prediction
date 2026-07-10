# -*- coding: UTF-8 -*-
"""
Build compact SGMF data files from the legacy 25-column CSV format.

Legacy rows repeat AS side information for every (userId, itemId, rating)
sample. This script splits that representation into:

1. one AS feature table:
   asId, info_type, AS_tier, ..., appearIXP, appearFac
2. one compact hop matrix per input file:
   userId, itemId, rating

Example:
    python build_compact_data.py ^
        --data_dir ./data/AShopmatrix ^
        --input_files train_hop_matrix_full.csv validate_hop_matrix_full.csv ^
        --output_dir ./data/AShopmatrix/compact
"""
import argparse
import os

import pandas as pd


FEATURE_NAMES = [
    "info_type",
    "AS_tier",
    "info_traffic",
    "info_ratio",
    "info_scope",
    "policy_general",
    "policy_locations",
    "policy_ratio",
    "policy_contracts",
    "appearIXP",
    "appearFac",
]

COLUMN_NAMES = (
    ["userId", "itemId", "rating"]
    + ["ASnode1_{}".format(name) for name in FEATURE_NAMES]
    + ["ASnode2_{}".format(name) for name in FEATURE_NAMES]
)


def read_legacy_csv(path, chunksize):
    return pd.read_csv(
        path,
        sep=r',(?=(?:[^"]*"[^"]*")*[^"]*$)',
        header=None,
        names=COLUMN_NAMES,
        engine='python',
        chunksize=chunksize)


def compact_name(input_name, suffix):
    base, ext = os.path.splitext(os.path.basename(input_name))
    return "{}{}{}".format(base, suffix, ext or ".csv")


def collect_feature_rows(chunk):
    frames = []
    for prefix, id_col in (("ASnode1", "userId"), ("ASnode2", "itemId")):
        cols = [id_col] + ["{}_{}".format(prefix, name) for name in FEATURE_NAMES]
        sub = chunk[cols].copy()
        sub.columns = ["asId"] + FEATURE_NAMES
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Build compact SGMF data files")
    parser.add_argument("--data_dir", default=".", help="Directory containing legacy input files")
    parser.add_argument("--input_files", nargs="+", required=True,
                        help="Legacy 25-column CSV files to convert")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory; defaults to data_dir")
    parser.add_argument("--feature_file", default="as_features.csv",
                        help="Output AS feature file name")
    parser.add_argument("--compact_suffix", default="_compact",
                        help="Suffix for compact hop matrix files")
    parser.add_argument("--chunksize", type=int, default=200000,
                        help="Rows per read chunk")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    feature_parts = []
    for input_file in args.input_files:
        input_path = input_file
        if not os.path.isabs(input_path):
            input_path = os.path.join(args.data_dir, input_file)
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)

        compact_path = os.path.join(output_dir, compact_name(input_file, args.compact_suffix))
        first_chunk = True
        rows_written = 0

        for chunk in read_legacy_csv(input_path, args.chunksize):
            chunk[["userId", "itemId", "rating"]].to_csv(
                compact_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False)
            feature_parts.append(collect_feature_rows(chunk))
            rows_written += len(chunk)
            first_chunk = False

        print("Wrote {} rows -> {}".format(rows_written, compact_path))

    if not feature_parts:
        raise ValueError("No input rows were read")

    features = pd.concat(feature_parts, ignore_index=True)
    before = len(features)
    features = features.drop_duplicates(subset=["asId"], keep="first")
    features = features.sort_values("asId").reset_index(drop=True)
    feature_path = os.path.join(output_dir, args.feature_file)
    features.to_csv(feature_path, index=False)

    print("Wrote {} AS feature rows -> {} (deduped from {})".format(
        len(features), feature_path, before))


if __name__ == "__main__":
    main()

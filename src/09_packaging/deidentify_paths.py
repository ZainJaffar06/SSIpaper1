#!/usr/bin/env python
"""Replace local image paths in published result files with a non-identifying image_id.

Some original export folder names embed dates of service (PHI under HIPAA Safe Harbor).
image_id = "img_" + first 16 hex chars of sha256(materialized_path). The mapping is
deterministic, so image_id joins identically across every published file; the
path-to-id mapping is written OUTSIDE the repository and is never committed.
Usage: python deidentify_paths.py <repo_results_dir> <offline_mapping_csv>"""
import sys, glob, hashlib, os
import pandas as pd

def image_id(path):
    return "img_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]

def main(results_dir, mapping_out):
    seen = {}
    for fp in sorted(glob.glob(os.path.join(results_dir, "**", "*.csv"), recursive=True)):
        df = pd.read_csv(fp, low_memory=False)
        if "materialized_path" not in df.columns:
            continue
        for p in df["materialized_path"].dropna().unique():
            seen[p] = image_id(p)
        df.insert(df.columns.get_loc("materialized_path"), "image_id", df["materialized_path"].map(image_id))
        df = df.drop(columns=["materialized_path"])
        df.to_csv(fp, index=False)
        print("de-identified:", os.path.relpath(fp, results_dir))
    pd.DataFrame({"materialized_path": list(seen), "image_id": list(seen.values())}).to_csv(mapping_out, index=False)
    print(f"{len(seen)} unique images mapped; mapping written offline to {mapping_out}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

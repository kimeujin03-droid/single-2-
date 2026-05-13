#!/usr/bin/env python3
from __future__ import annotations

import argparse

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.runner import run_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-null", type=int, default=None)
    ap.add_argument("--max-backgrounds", type=int, default=None)
    ap.add_argument("--no-save-null", action="store_true")
    args = ap.parse_args()
    csv_path = run_grid(args.config, args.out, n_null=args.n_null, max_backgrounds=args.max_backgrounds, save_null=not args.no_save_null)
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathb.runner import benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-null", type=int, default=5)
    args = ap.parse_args()
    info = benchmark(args.config, args.out, n_null=args.n_null)
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

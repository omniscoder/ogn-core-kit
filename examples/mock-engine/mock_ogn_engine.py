#!/usr/bin/env python3
from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    if "--version" in argv:
        print("ogn_run version 0.0.0-mock (git deadbeef)")
        return 0

    try:
        out_index = argv.index("--vcf")
    except ValueError:
        print("missing --vcf", file=sys.stderr)
        return 2

    if out_index + 1 >= len(argv):
        print("missing --vcf value", file=sys.stderr)
        return 2

    out_path = argv[out_index + 1]
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("##fileformat=VCFv4.3\n")
        fp.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n")
        fp.write("chr20\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

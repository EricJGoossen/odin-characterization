"""Small helpers.

    python -m odin_char.cli inspect <bag>     topics, message counts and tf pairs in a bag
"""
from __future__ import annotations

import argparse

from .bagio import list_bag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect", help="list topics and tf pairs in a bag")
    p.add_argument("bag")
    args = ap.parse_args()
    if args.cmd == "inspect":
        print(list_bag(args.bag))


if __name__ == "__main__":
    main()

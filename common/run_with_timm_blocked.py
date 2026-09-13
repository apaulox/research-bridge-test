#!/usr/bin/env python3
"""Run a script while failing immediately if it imports timm."""

import argparse
import importlib.abc
import runpy
import sys


class BlockTimm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "timm" or fullname.startswith("timm."):
            raise ModuleNotFoundError("timm intentionally blocked for DACON test")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.meta_path.insert(0, BlockTimm())
    sys.argv = [args.script] + args.script_args
    runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()

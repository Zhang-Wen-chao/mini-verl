"""Convert downloaded public GSM8K JSONL files to the first-run task schema."""

from __future__ import annotations

import argparse

from gsm8k import convert_records, read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source GSM8K JSONL with question and answer fields")
    parser.add_argument("--output", required=True, help="Output JSONL containing prompt and answer fields")
    parser.add_argument("--limit", type=int, help="Optional positive row limit for a smoke run")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    from pathlib import Path

    count = write_jsonl(convert_records(read_jsonl(Path(args.input))), Path(args.output), args.limit)
    print(f"wrote {count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

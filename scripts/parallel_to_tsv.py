from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.malaycn.text import normalize_text


def parse_args():
    parser = argparse.ArgumentParser(description="把两个逐行对齐的文本文件转换为训练用 TSV")
    parser.add_argument("--source-file", required=True, help="马来语文本，每行一句")
    parser.add_argument("--target-file", required=True, help="中文文本，每行一句")
    parser.add_argument("--output-file", required=True, help="输出 TSV 文件")
    parser.add_argument("--max-source-chars", type=int, default=300)
    parser.add_argument("--max-target-chars", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = Path(args.source_file)
    target_path = Path(args.target_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with source_path.open("r", encoding="utf-8") as source_f, target_path.open(
        "r", encoding="utf-8"
    ) as target_f, output_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["ms", "zh"])
        for source, target in zip(source_f, target_f):
            raw_source = source.strip()
            raw_target = target.strip()
            if not raw_source or not raw_target:
                continue
            if len(raw_source) > args.max_source_chars or len(raw_target) > args.max_target_chars:
                continue
            if not normalize_text(raw_source, "ms") or not normalize_text(raw_target, "zh"):
                continue
            writer.writerow([raw_source, raw_target])
            written += 1

    print(f"已写入 {written} 条句对: {output_path}")


if __name__ == "__main__":
    main()


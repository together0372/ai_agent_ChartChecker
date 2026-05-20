import sys
import os
import random
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from detector import detect_from_file

CHARTS_DIR = Path(__file__).parent.parent / "raw_data" / "charts"


def test_folder(folder_path: Path, n: int = 20):
    files = list(folder_path.glob("*.jpg"))
    if not files:
        print(f"  [skip] 파일 없음")
        return 0, 0

    samples = random.sample(files, min(n, len(files)))
    detected = 0
    for f in samples:
        result = detect_from_file(str(f))
        is_chart = result.get("is_chart", False)
        if is_chart:
            detected += 1
        status = "O" if is_chart else "X"
        score = result.get("score", 0)
        h = result.get("h_lines", 0)
        v = result.get("v_lines", 0)
        r = result.get("rects", 0)
        c = result.get("circles", 0)
        print(f"  [{status}] score={score}  h={h} v={v} rect={r} circ={c}  {f.name}")

    return detected, len(samples)


def main(n: int = 20):
    print(f"\n=== 룰베이스 디텍터 테스트 (threshold=3) ===\n")

    total_detected = 0
    total_samples = 0

    for folder in sorted(CHARTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        print(f"[{folder.name}]")
        t0 = time.perf_counter()
        detected, cnt = test_folder(folder, n=n)
        elapsed = (time.perf_counter() - t0) * 1000
        total_detected += detected
        total_samples += cnt
        print(f"  -> {detected}/{cnt} 감지 ({detected/cnt*100:.0f}%)  {elapsed:.0f}ms\n")

    print(f"=== 전체: {total_detected}/{total_samples} 감지 ({total_detected/total_samples*100:.0f}%) ===")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None)
    ap.add_argument("--n",   type=int, default=20)
    args = ap.parse_args()

    if args.dir:
        folder = Path(args.dir)
        print(f"\n=== 룰베이스 디텍터 테스트 ===\n[{folder.name}]")
        detected, n = test_folder(folder, n=args.n)
        if n:
            print(f"  -> {detected}/{n} 감지 ({detected/n*100:.0f}%)")
    else:
        main(n=args.n)

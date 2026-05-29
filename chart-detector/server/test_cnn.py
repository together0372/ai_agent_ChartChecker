import sys
import os
import random
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from cnn import _get_model, classify_image, THRESHOLD

CHARTS_DIR = Path(__file__).parent.parent / "raw_data" / "charts"


def test_folder(folder_path: Path, n: int = 20):
    files = list(folder_path.glob("*.jpg"))
    if not files:
        print(f"  [skip] 파일 없음")
        return 0, 0

    samples = random.sample(files, min(n, len(files)))
    detected = 0
    for f in samples:
        result = classify_image(str(f))
        if result["is_chart"]:
            detected += 1
        status = "O" if result["is_chart"] else "X"
        print(f"  [{status}] conf={result['confidence']:.3f}  {f.name}")

    return detected, len(samples)


def main(n: int = 20):
    print(f"\n=== 차트 분류기 테스트 (threshold={THRESHOLD}) ===\n")

    t0 = time.perf_counter()
    _get_model()
    print(f"모델 로드: {(time.perf_counter()-t0)*1000:.0f}ms\n")

    total_detected = 0
    total_samples = 0

    for folder in sorted(CHARTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        print(f"[{folder.name}]")
        detected, cnt = test_folder(folder, n=n)
        total_detected += detected
        total_samples += cnt
        print(f"  -> {detected}/{cnt} 감지 ({detected/cnt*100:.0f}%)\n")

    print(f"=== 전체: {total_detected}/{total_samples} 감지 ({total_detected/total_samples*100:.0f}%) ===")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None, help="테스트할 폴더 (기본: raw_data/charts 전체)")
    ap.add_argument("--n",   type=int, default=20,   help="폴더당 샘플 수")
    args = ap.parse_args()

    if args.dir:
        folder = Path(args.dir)
        print(f"\n=== 차트 분류기 테스트 (threshold={THRESHOLD}) ===\n")
        t0 = time.perf_counter()
        _get_model()
        print(f"모델 로드: {(time.perf_counter()-t0)*1000:.0f}ms\n")
        print(f"[{folder.name}]")
        detected, n = test_folder(folder, n=args.n)
        if n:
            print(f"  -> {detected}/{n} 감지 ({detected/n*100:.0f}%)")
    else:
        main(n=args.n)

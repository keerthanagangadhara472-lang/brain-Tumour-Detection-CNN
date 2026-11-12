import argparse, random, shutil
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def main():
    p = argparse.ArgumentParser(description="Split class folders into train/test folders.")
    p.add_argument("--source", default="data/all", help="Source containing no_tumor/ and tumor/")
    p.add_argument("--output", default="data", help="Output root")
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    src, out = Path(args.source), Path(args.output)
    random.seed(args.seed)
    for cls in ("no_tumor", "tumor"):
        files = [f for f in (src / cls).glob("*") if f.suffix.lower() in EXTS]
        if not files:
            raise FileNotFoundError(f"No images found in {src / cls}")
        random.shuffle(files)
        n_test = max(1, int(len(files) * args.test_ratio))
        for split, subset in (("test", files[:n_test]), ("train", files[n_test:])):
            target = out / split / cls
            target.mkdir(parents=True, exist_ok=True)
            for f in subset:
                shutil.copy2(f, target / f.name)
        print(f"{cls}: {len(files)-n_test} train, {n_test} test")
    print("Split completed. Review patient-level leakage if patient IDs exist.")

if __name__ == "__main__":
    main()

import argparse
import shutil
from pathlib import Path

from src.mockturtle import cut_aig, rewrite_aig, aig_to_tt

def run_cut_aig(aig_path: Path, out_base: Path, cut_size: int) -> int:
    name = aig_path.stem
    aig_dir = out_base / str(cut_size) / "aig"
    if aig_dir.exists():
        shutil.rmtree(aig_dir)
    aig_dir.mkdir(parents=True)

    n = cut_aig(str(aig_path), str(aig_dir), cut_size)
    print(f"[{name}] {n} subcircuits")
    return n

def run_rewrite_aig(out_base: Path, cut_size: int) -> int:
    name = out_base.name
    aig_dir = out_base / str(cut_size) / "aig"
    if not aig_dir.is_dir():
        print(f"[{name}] aig/ not found — run cut_aig first")
        return 0

    out_dir = out_base / str(cut_size) / "aig_rewriting"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    aig_files = list(aig_dir.glob("*.aig"))
    for aig_path in aig_files:
        rewrite_aig(str(aig_path), str(out_dir / aig_path.name))

    n = len(aig_files)
    print(f"[{name}] {n} rewritten")
    return n

def run_simulate_tt(out_base: Path, cut_size: int) -> int:
    name = out_base.name
    aig_dir = out_base / str(cut_size) / "aig"
    if not aig_dir.is_dir():
        print(f"[{name}] aig/ not found — run cut_aig first")
        return 0

    truth_dir = out_base / str(cut_size) / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)

    aig_files = list(aig_dir.glob("*.aig"))
    for aig_path in aig_files:
        tt = aig_to_tt(str(aig_path))
        (truth_dir / f"{aig_path.stem}.txt").write_text(tt + "\n")

    n = len(aig_files)
    print(f"[{name}] {n} truth tables simulated")
    return n

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=["cut_aig", "rewrite_aig", "simulate_tt"],
        help="Which step to run",
    )
    parser.add_argument(
        "--data-dir",
        default="third-party/benchmarks",
        help="Directory containing flat <name>.aig benchmark files; only used "
             "by cut_aig (default: third-party/benchmarks)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks",
        help="Root directory for prepared data, written as "
             "<output-dir>/<name>/<cut_size>/ (default: benchmarks)",
    )
    parser.add_argument(
        "--cut-size",
        type=int,
        default=6,
        help="Number of leaves per reconvergence-driven cut (default: 6)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.command == "cut_aig":
        data_dir = Path(args.data_dir)
        if not data_dir.is_dir():
            raise SystemExit(f"Data directory not found: {data_dir}")

        aig_files = sorted(data_dir.rglob("*.aig"))
        if not aig_files:
            raise SystemExit(f"No .aig benchmarks found in {data_dir}")

        total = 0
        for aig_path in aig_files:
            total += run_cut_aig(aig_path, output_dir / aig_path.stem, args.cut_size)
        print(f"\nTotal {args.cut_size}-bit cuts: {total}\n")

    elif args.command == "rewrite_aig":
        if not output_dir.is_dir():
            raise SystemExit(f"Output directory not found: {output_dir}")

        bench_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir())
        if not bench_dirs:
            raise SystemExit(f"No prepared benchmark dirs found in {output_dir}")

        total = 0
        for out_base in bench_dirs:
            total += run_rewrite_aig(out_base, args.cut_size)
        print(f"\nTotal {args.cut_size}-bit cuts rewritten: {total}\n")

    elif args.command == "simulate_tt":
        if not output_dir.is_dir():
            raise SystemExit(f"Output directory not found: {output_dir}")

        bench_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir())
        if not bench_dirs:
            raise SystemExit(f"No prepared benchmark dirs found in {output_dir}")

        total = 0
        for out_base in bench_dirs:
            total += run_simulate_tt(out_base, args.cut_size)
        print(f"\nTotal {args.cut_size}-bit cuts simulated: {total}\n")

    else:
        raise SystemExit(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()

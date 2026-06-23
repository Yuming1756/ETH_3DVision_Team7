from pathlib import Path
import os
import subprocess

def du(path: Path) -> str:
    try:
        return subprocess.check_output(["du", "-sh", str(path)], text=True).split()[0]
    except Exception:
        return "unknown"

roots = []

for value in [
    os.environ.get("HF_HOME"),
    os.environ.get("HUGGINGFACE_HUB_CACHE"),
    os.environ.get("TRANSFORMERS_CACHE"),
    "/work/courses/3dv/team7/.cache/huggingface",
    str(Path.home() / ".cache" / "huggingface"),
]:
    if value:
        p = Path(value).expanduser()
        if p.exists() and p not in roots:
            roots.append(p)

patterns = [
    "models--Qwen*",
    "models--depth-anything*",
]

print("HuggingFace model snapshot manifest")
print("Generated from local cache scan")
print()

if not roots:
    print("No HuggingFace cache roots found.")
    raise SystemExit(0)

for root in roots:
    print("=" * 90)
    print(f"CACHE ROOT: {root}")
    print(f"SIZE: {du(root)}")
    print()

    candidate_bases = [root]
    if (root / "hub").exists():
        candidate_bases.append(root / "hub")

    found_any = False

    for base in candidate_bases:
        if not base.exists():
            continue

        model_dirs = []
        for pat in patterns:
            model_dirs.extend(sorted(base.glob(pat)))

        for model_dir in model_dirs:
            if not model_dir.is_dir():
                continue

            found_any = True
            print("-" * 90)
            print(f"MODEL CACHE DIR: {model_dir}")
            print(f"MODEL CACHE SIZE: {du(model_dir)}")

            refs_dir = model_dir / "refs"
            if refs_dir.exists():
                print("REFS:")
                for ref_file in sorted(refs_dir.iterdir()):
                    if ref_file.is_file():
                        try:
                            print(f"  {ref_file.name}: {ref_file.read_text().strip()}")
                        except Exception:
                            print(f"  {ref_file.name}: <could not read>")
            else:
                print("REFS: none")

            snapshots_dir = model_dir / "snapshots"
            if snapshots_dir.exists():
                print("SNAPSHOTS:")
                snapshots = [p for p in sorted(snapshots_dir.iterdir()) if p.is_dir()]
                if not snapshots:
                    print("  none")
                for snap in snapshots:
                    print(f"  snapshot: {snap.name}")
                    print(f"  path: {snap}")
                    print(f"  size: {du(snap)}")

                    config = snap / "config.json"
                    processor = snap / "preprocessor_config.json"
                    tokenizer = snap / "tokenizer_config.json"

                    if config.exists():
                        print(f"  has config.json: yes")
                    if processor.exists():
                        print(f"  has preprocessor_config.json: yes")
                    if tokenizer.exists():
                        print(f"  has tokenizer_config.json: yes")

                    weight_files = (
                        list(snap.glob("*.safetensors"))
                        + list(snap.glob("*.bin"))
                        + list(snap.glob("*.pt"))
                        + list(snap.glob("*.pth"))
                    )
                    print(f"  weight files: {len(weight_files)}")
                    for wf in sorted(weight_files)[:20]:
                        print(f"    {wf.name} ({du(wf)})")
                    if len(weight_files) > 20:
                        print(f"    ... {len(weight_files) - 20} more")
            else:
                print("SNAPSHOTS: none")

            print()

    if not found_any:
        print("No Qwen or depth-anything model caches found under this root.")
        print()

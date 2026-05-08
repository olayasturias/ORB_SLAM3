#!/usr/bin/env python3
"""
Run multiple ORB-SLAM3 trajectories as a batch and save in eiva-slam-traj-eval format.

Batch structure:
  <sandbox_root>/<batch_date>/
    children.txt                              — paths relative to sandbox_root
    orb-slam3@<project_name>/<run_date>/
      poses.npy
      ref_poses.npy
      config.yaml
      metadata.yaml

Config YAML format:
  runs:
    - project: my_project
      ref_poses: path/to/gt.txt   # optional
      exe_args:
        - Examples\\Monocular\\Release\\mono_eiva.exe
        - Vocabulary\\ORBvoc.txt
        - Examples\\Monocular\\EIVA.yaml
        - D:\\Datasets\\EIVA\\plane_nose

Usage:
    python run_batch.py [--sandbox-root PATH] <config.yaml>
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from run_sandbox import (
    _git_version,
    _txt_to_npy,
    _GT_LOADERS,
    _resolve_existing,
    _FULL_TRAJ_NAMES,
    _KF_TRAJ_NAMES,
)

_DEFAULT_SANDBOX_ROOT = Path(__file__).resolve().parent / "runs"


def _run_one(folder: Path, project_name: str, exe_args: list, ref_poses_path: str | None) -> int:
    slam_yaml = exe_args[2] if len(exe_args) > 2 else None

    with open(folder / "config.yaml", "w") as f:
        cfg: dict = {"Project": project_name}
        if slam_yaml:
            cfg["slam_config"] = slam_yaml
        yaml.dump(cfg, f, default_flow_style=False)

    if slam_yaml and Path(slam_yaml).exists():
        shutil.copy(slam_yaml, folder / "slam_config.yaml")

    meta = {
        "time": folder.name,
        "git_version": _git_version(Path(__file__).parent),
        "command": " ".join(sys.argv),
    }
    with open(folder / "metadata.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    abs_args = [_resolve_existing(a) for a in exe_args]
    print(f"  Command : {' '.join(abs_args)}\n")
    result = subprocess.run(abs_args, cwd=str(folder))

    print("  Converting trajectories …")
    has_full = False
    for name in _FULL_TRAJ_NAMES:
        src = folder / name
        if src.exists():
            if _txt_to_npy(src, folder / "poses.npy"):
                has_full = True
            break
    for name in _KF_TRAJ_NAMES:
        src = folder / name
        if src.exists():
            dst_name = "kf_poses.npy" if has_full else "poses.npy"
            _txt_to_npy(src, folder / dst_name)
            break
    for f_txt in sorted(folder.glob("f_*.txt")):
        base = f_txt.stem[2:]
        dst_name = f"poses_{base}.npy" if has_full else "poses.npy"
        if _txt_to_npy(f_txt, folder / dst_name):
            has_full = True
    for kf_txt in sorted(folder.glob("kf_*.txt")):
        base = kf_txt.stem[3:]
        _txt_to_npy(kf_txt, folder / f"kf_poses_{base}.npy")

    if not (folder / "poses.npy").exists():
        print("  Warning: poses.npy not created.", file=sys.stderr)

    print("  Loading reference poses …")
    ref_saved = False
    if ref_poses_path:
        ref_src = Path(ref_poses_path)
        if ref_src.suffix == ".npy":
            shutil.copy(ref_src, folder / "ref_poses.npy")
            print(f"    Copied {ref_src.name} -> ref_poses.npy")
            ref_saved = True
        elif ref_src.exists():
            ref_saved = _txt_to_npy(ref_src, folder / "ref_poses.npy")
        else:
            print(f"    Warning: ref_poses not found: {ref_src}", file=sys.stderr)
    else:
        loader = _GT_LOADERS.get(Path(exe_args[0]).stem)
        if loader:
            ref_data = loader(exe_args)
            if ref_data is not None and len(ref_data) > 0:
                np.save(str(folder / "ref_poses.npy"), ref_data)
                print(f"    ref_poses.npy  shape={ref_data.shape}")
                ref_saved = True
            else:
                print("    Warning: GT loader returned no data.", file=sys.stderr)
        else:
            print(f"    No GT loader for '{Path(exe_args[0]).stem}' — skipping ref_poses.")

    if not ref_saved:
        print("    ref_poses.npy not saved.")

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run multiple ORB-SLAM3 trajectories as a batch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sandbox-root", type=Path, default=_DEFAULT_SANDBOX_ROOT)
    parser.add_argument("config", type=Path, help="YAML config listing runs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    runs = cfg.get("runs", [])
    if not runs:
        print("No runs defined in config.", file=sys.stderr)
        return 1

    batch_ts = datetime.now().strftime("%m_%d_%H%M%S")
    batch_folder = args.sandbox_root / batch_ts
    batch_folder.mkdir(parents=True, exist_ok=True)
    print(f"Batch folder : {batch_folder}")
    print(f"Runs         : {len(runs)}\n")

    child_paths: list[str] = []
    exit_codes: list[int] = []

    for i, run in enumerate(runs, 1):
        project_name = run["project"]
        exe_args = [str(a) for a in run["exe_args"]]
        ref_poses = run.get("ref_poses")

        run_ts = datetime.now().strftime("%m_%d_%H%M%S")
        run_folder = batch_folder / f"orb-slam3@{project_name}" / run_ts
        run_folder.mkdir(parents=True, exist_ok=True)

        print(f"{'='*60}")
        print(f"[{i}/{len(runs)}] {project_name}")
        print(f"  Folder : {run_folder}")

        code = _run_one(run_folder, project_name, exe_args, ref_poses)
        exit_codes.append(code)

        rel = run_folder.relative_to(args.sandbox_root)
        child_paths.append(str(rel).replace("\\", "/"))

    children_file = batch_folder / "children.txt"
    with open(children_file, "w") as f:
        f.write("\n".join(child_paths) + "\n")

    n_ok = sum(c == 0 for c in exit_codes)
    print(f"\n{'='*60}")
    print(f"Batch complete: {n_ok}/{len(runs)} runs succeeded")
    print(f"Sandbox      : {batch_folder}")
    return 0 if all(c == 0 for c in exit_codes) else 1


if __name__ == "__main__":
    sys.exit(main())

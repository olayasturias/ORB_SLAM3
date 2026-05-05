#!/usr/bin/env python3
"""
Run an ORB-SLAM3 executable and save the experiment in sandbox format.

The sandbox format (compatible with eiva-slam-traj-eval) organises each run
in a timestamped subfolder and stores poses.npy (N x 8: timestamp, tx, ty,
tz, qx, qy, qz, qw) alongside metadata / config so evaluation scripts can
consume it directly.

Usage:
    python run_sandbox.py [--sandbox-root PATH] [--project NAME] -- <exe> [exe args ...]

Example (from ORB_SLAM3 root on Windows):
    python run_sandbox.py --project mimir_seafloor -- ^
        build_msvc\\Release\\mono_mimir.exe ^
        Vocabulary/ORBvoc.txt ^
        Examples/Monocular/MIMIR.yaml ^
        /data/MIMIR/SeaFloor/track0 ^
        Examples/Monocular/MIMIR_Times.txt
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

_DEFAULT_SANDBOX_ROOT = Path(__file__).resolve().parent / "runs"

# Ordered preference for the full-trajectory file produced by ORB-SLAM3.
_FULL_TRAJ_NAMES = ["CameraTrajectory.txt"]
_KF_TRAJ_NAMES   = ["KeyFrameTrajectory.txt"]


# ---------------------------------------------------------------------------
# Minimal sandbox helpers — mirrors the structure of utility/sandbox.py from
# eiva-slam-traj-eval without importing its heavy dependencies (yacs, torch…).
# ---------------------------------------------------------------------------

def _get_git_version(cwd: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(cwd),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "NOT_AVAILABLE"


def create_sandbox(root: Path, project_name: str) -> Path:
    """Create a sandbox folder and write metadata.yaml; return the folder path."""
    time_str = datetime.now().strftime("%m_%d_%H%M%S")
    folder = root / project_name / time_str
    folder.mkdir(parents=True, exist_ok=True)

    meta = {
        "time": time_str,
        "git_version": _get_git_version(Path(__file__).parent),
        "command": " ".join(sys.orig_argv),
    }
    with open(folder / "metadata.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    return folder


# ---------------------------------------------------------------------------

def _resolve_existing(arg: str) -> str:
    """Return absolute path if arg points to an existing file/dir, else unchanged."""
    p = Path(arg)
    if p.exists():
        return str(p.resolve())
    return arg


def _txt_to_npy(src: Path, dst: Path) -> bool:
    """Load a whitespace-separated trajectory txt and save as float64 .npy."""
    try:
        data = np.loadtxt(str(src), dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        np.save(str(dst), data)
        print(f"  {src.name} -> {dst.name}  shape={data.shape}")
        return True
    except Exception as exc:
        print(f"  Warning: could not convert {src.name}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ORB-SLAM3 and save the experiment in sandbox format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sandbox-root", type=Path, default=_DEFAULT_SANDBOX_ROOT,
        help="Parent directory for sandbox runs (default: ./runs)",
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="Project name stored in config.yaml / used in evaluation tables.",
    )
    parser.add_argument("exe_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    exe_args = args.exe_args
    if exe_args and exe_args[0] == "--":
        exe_args = exe_args[1:]

    if not exe_args:
        parser.print_help()
        return 1

    executable = exe_args[0]
    project_name = args.project or Path(executable).stem

    # ------------------------------------------------------------------ sandbox
    folder = create_sandbox(args.sandbox_root, project_name)
    print(f"Sandbox : {folder}")

    # Write config.yaml — 'Project' is required by the eval framework.
    slam_yaml = exe_args[2] if len(exe_args) > 2 else None
    cfg_data: dict = {"Project": project_name}
    if slam_yaml:
        cfg_data["slam_config"] = slam_yaml
    with open(folder / "config.yaml", "w") as f:
        yaml.dump(cfg_data, f, default_flow_style=False)

    # Keep a copy of the SLAM calibration yaml for reproducibility.
    if slam_yaml and Path(slam_yaml).exists():
        shutil.copy(slam_yaml, folder / "slam_config.yaml")

    # ---------------------------------------------------------- run executable
    # Resolve relative paths to absolute so running from the sandbox folder
    # doesn't break dataset / vocab / config lookups.
    abs_args = [_resolve_existing(a) for a in exe_args]

    print(f"Command : {' '.join(abs_args)}\n")
    result = subprocess.run(abs_args, cwd=str(folder))

    # ------------------------------------------------ convert trajectories
    print("\nConverting trajectories …")
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

    # Handle named outputs from mono_mimir: f_<name>.txt / kf_<name>.txt
    for f_txt in sorted(folder.glob("f_*.txt")):
        base = f_txt.stem[2:]  # strip "f_" prefix
        dst_name = f"poses_{base}.npy" if has_full else "poses.npy"
        if _txt_to_npy(f_txt, folder / dst_name):
            has_full = True

    for kf_txt in sorted(folder.glob("kf_*.txt")):
        base = kf_txt.stem[3:]  # strip "kf_" prefix
        _txt_to_npy(kf_txt, folder / f"kf_poses_{base}.npy")

    if not (folder / "poses.npy").exists():
        print(
            "Warning: poses.npy was not created — no recognised trajectory file found.",
            file=sys.stderr,
        )

    print(f"\nDone. Sandbox at:\n  {folder}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

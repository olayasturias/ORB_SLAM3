#!/usr/bin/env python3
"""
Run an ORB-SLAM3 executable and save the experiment in sandbox format.

The sandbox format (compatible with eiva-slam-traj-eval) organises each run
in a timestamped subfolder and stores:
  poses.npy     — estimated trajectory  (N x 8: timestamp, tx, ty, tz, qx, qy, qz, qw)
  ref_poses.npy — reference / GT poses  (same format, auto-loaded per dataset)

Supported datasets for automatic GT loading:
  mono_aqualoc / stereo_aqualoc      — COLMAP-based GT
  mono_subpipe / mono_inertial_subpipe — EstimatedState.csv
  mono_euroc / stereo_euroc / *_inertial_euroc — EuRoC ASL CSV

For other datasets pass --ref-poses <file> (txt or npy).

Usage:
    python run_sandbox.py [--sandbox-root PATH] [--project NAME]
                          [--ref-poses FILE]
                          -- <exe> [exe args ...]

Example (from ORB_SLAM3 root on Windows):
    python run_sandbox.py --project aqualoc_arch_1 -- ^
        Examples\\Monocular\\Release\\mono_aqualoc.exe ^
        Vocabulary\\ORBvoc.txt ^
        Examples\\Monocular\\Aqualoc.yaml ^
        D:\\Datasets\\Aqualoc\\Archaeological_site_sequences 1
"""

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

_DEFAULT_SANDBOX_ROOT = Path(__file__).resolve().parent / "runs"

_FULL_TRAJ_NAMES = ["CameraTrajectory.txt"]
_KF_TRAJ_NAMES   = ["KeyFrameTrajectory.txt"]


# ---------------------------------------------------------------------------
# Minimal sandbox creation
# ---------------------------------------------------------------------------

def _git_version(cwd: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(cwd), stderr=subprocess.DEVNULL,
            ).decode().strip()
        )
    except Exception:
        return "NOT_AVAILABLE"


def _create_sandbox(root: Path, project_name: str) -> Path:
    time_str = datetime.now().strftime("%m_%d_%H%M%S")
    folder = root / project_name / time_str
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "time": time_str,
        "git_version": _git_version(Path(__file__).parent),
        "command": " ".join(sys.orig_argv),
    }
    with open(folder / "metadata.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)
    return folder


# ---------------------------------------------------------------------------
# Trajectory conversion helpers
# ---------------------------------------------------------------------------

def _resolve_existing(arg: str) -> str:
    p = Path(arg)
    return str(p.resolve()) if p.exists() else arg


def _txt_to_npy(src: Path, dst: Path) -> bool:
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


# ---------------------------------------------------------------------------
# Ground-truth loaders  (return (N, 8) float64 or None)
#
# Output column order: [timestamp_s, tx, ty, tz, qx, qy, qz, qw]
# ---------------------------------------------------------------------------

def _load_ref_subpipe(exe_args: list) -> "np.ndarray | None":
    """
    GT from EstimatedState.csv (same file used to load images).
    col 1  = timestamp
    cols 7-9  = x, y, z
    cols 10-13 = qx, qy, qz, qw
    """
    if len(exe_args) < 4:
        return None
    csv_file = Path(exe_args[3]) / "EstimatedState.csv"
    if not csv_file.exists():
        print(f"  GT not found: {csv_file}", file=sys.stderr)
        return None

    rows = []
    with open(csv_file) as f:
        next(f)  # skip header
        for line in f:
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) < 14:
                continue
            try:
                rows.append([
                    float(parts[1]),                                  # timestamp
                    float(parts[7]), float(parts[8]), float(parts[9]),  # x y z
                    float(parts[10]), float(parts[11]),               # qx qy
                    float(parts[12]), float(parts[13]),               # qz qw
                ])
            except (ValueError, IndexError):
                continue

    return np.array(rows, dtype=np.float64) if rows else None


def _load_ref_aqualoc(exe_args: list) -> "np.ndarray | None":
    """
    GT file: <dataset>/../{archaeo,harbor}_groundtruth_files/new_*_colmap_traj_sequence_NN.txt
    Line format: frameNNNNNN  px py pz  qx qy qz qw

    Frame→timestamp via: <dataset>/raw_data/img_sequence_N.csv
    CSV col 0 = timestamp_ns, col 1 = image filename containing 'frameNNNNNN'
    Timestamps converted to seconds to match ORB-SLAM3 output.
    """
    if len(exe_args) < 5:
        return None

    dataset_path = Path(exe_args[3])
    seq_num = exe_args[4]
    seq_int = int(seq_num)

    gt_candidates = [
        dataset_path.parent / "archaeo_groundtruth_files"
            / f"new_archaeo_colmap_traj_sequence_{seq_int:02d}.txt",
        dataset_path.parent / "harbor_groundtruth_files"
            / f"new_harbor_colmap_traj_sequence_{seq_int:02d}.txt",
        dataset_path / "raw_data"
            / f"new_archaeo_colmap_traj_sequence_{seq_num}.txt",
        dataset_path / "raw_data"
            / f"new_harbor_colmap_traj_sequence_{seq_num}.txt",
    ]
    gt_file = next((p for p in gt_candidates if p.exists()), None)
    if gt_file is None:
        print(f"  GT not found (tried {len(gt_candidates)} paths)", file=sys.stderr)
        return None

    # Build frame -> timestamp_s map
    csv_candidates = [
        dataset_path / "raw_data" / f"img_sequence_{seq_num}.csv",
        dataset_path.parent / f"img_sequence_{seq_int:02d}.csv",
        dataset_path.parent / f"harbor_img_sequence_{seq_int:02d}.csv",
    ]
    img_csv = next((p for p in csv_candidates if p.exists()), None)

    frame_to_ts: dict[int, float] = {}
    if img_csv:
        with open(img_csv) as f:
            next(f, None)  # skip header if present
            for line in f:
                parts = [p.strip() for p in line.strip().split(',')]
                if len(parts) < 2:
                    continue
                try:
                    ts_s = float(parts[0]) / 1e9
                    m = re.search(r'frame(\d+)', parts[1])
                    if m:
                        frame_to_ts[int(m.group(1))] = ts_s
                except (ValueError, IndexError):
                    continue

    rows = []
    with open(gt_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                m = re.search(r'(\d+)', parts[0])
                frame_num = int(m.group(1)) if m else None
                px, py, pz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                if frame_num is not None and frame_num in frame_to_ts:
                    ts = frame_to_ts[frame_num]
                elif frame_to_ts:
                    continue  # have a map but this frame isn't in it
                else:
                    ts = float(frame_num) if frame_num is not None else 0.0

                rows.append([ts, px, py, pz, qx, qy, qz, qw])
            except (ValueError, IndexError, AttributeError):
                continue

    return np.array(rows, dtype=np.float64) if rows else None


def _load_ref_euroc(exe_args: list) -> "np.ndarray | None":
    """
    GT: <sequence_folder>/mav0/state_groundtruth_estimate0/data.csv
    CSV header + rows: timestamp_ns, px, py, pz, qw, qx, qy, qz, ...
    Returns timestamps in seconds; quaternion reordered to xyzw.
    """
    if len(exe_args) < 4:
        return None
    gt_file = (
        Path(exe_args[3]) / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    )
    if not gt_file.exists():
        print(f"  GT not found: {gt_file}", file=sys.stderr)
        return None

    data = np.loadtxt(str(gt_file), delimiter=',', skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    timestamps = data[:, 0] / 1e9           # ns → s
    positions  = data[:, 1:4]               # px py pz
    q_xyzw     = np.roll(data[:, 4:8], -1, axis=1)  # wxyz → xyzw

    return np.column_stack([timestamps, positions, q_xyzw])


# Map executable stem -> GT loader function
_GT_LOADERS = {
    "mono_aqualoc":           _load_ref_aqualoc,
    "stereo_aqualoc":         _load_ref_aqualoc,
    "mono_subpipe":           _load_ref_subpipe,
    "mono_inertial_subpipe":  _load_ref_subpipe,
    "mono_euroc":             _load_ref_euroc,
    "stereo_euroc":           _load_ref_euroc,
    "mono_inertial_euroc":    _load_ref_euroc,
    "stereo_inertial_euroc":  _load_ref_euroc,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument(
        "--ref-poses", type=str, default=None, metavar="FILE",
        help="Path to reference / ground-truth poses (.npy or whitespace-sep txt). "
             "Auto-detected for Aqualoc, SubPipe, EuRoC when omitted.",
    )
    parser.add_argument("exe_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    exe_args = args.exe_args
    if exe_args and exe_args[0] == "--":
        exe_args = exe_args[1:]

    if not exe_args:
        parser.print_help()
        return 1

    executable  = exe_args[0]
    project_name = args.project or Path(executable).stem

    # ------------------------------------------------------------------ sandbox
    folder = _create_sandbox(args.sandbox_root, project_name)
    print(f"Sandbox : {folder}")

    slam_yaml = exe_args[2] if len(exe_args) > 2 else None
    with open(folder / "config.yaml", "w") as f:
        cfg: dict = {"Project": project_name}
        if slam_yaml:
            cfg["slam_config"] = slam_yaml
        yaml.dump(cfg, f, default_flow_style=False)

    if slam_yaml and Path(slam_yaml).exists():
        shutil.copy(slam_yaml, folder / "slam_config.yaml")

    # ---------------------------------------------------------- run executable
    abs_args = [_resolve_existing(a) for a in exe_args]
    print(f"Command : {' '.join(abs_args)}\n")
    result = subprocess.run(abs_args, cwd=str(folder))

    # ------------------------------------------------ convert estimated poses
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

    for f_txt in sorted(folder.glob("f_*.txt")):
        base = f_txt.stem[2:]
        dst_name = f"poses_{base}.npy" if has_full else "poses.npy"
        if _txt_to_npy(f_txt, folder / dst_name):
            has_full = True

    for kf_txt in sorted(folder.glob("kf_*.txt")):
        base = kf_txt.stem[3:]
        _txt_to_npy(kf_txt, folder / f"kf_poses_{base}.npy")

    if not (folder / "poses.npy").exists():
        print("  Warning: poses.npy not created — no recognised trajectory file found.",
              file=sys.stderr)

    # ------------------------------------------------- save reference poses
    print("\nLoading reference poses …")
    ref_saved = False

    if args.ref_poses:
        ref_src = Path(args.ref_poses)
        if ref_src.suffix == ".npy":
            shutil.copy(ref_src, folder / "ref_poses.npy")
            print(f"  Copied {ref_src.name} -> ref_poses.npy")
            ref_saved = True
        elif ref_src.exists():
            ref_saved = _txt_to_npy(ref_src, folder / "ref_poses.npy")
        else:
            print(f"  Warning: --ref-poses file not found: {ref_src}", file=sys.stderr)
    else:
        loader = _GT_LOADERS.get(Path(executable).stem)
        if loader:
            ref_data = loader(exe_args)
            if ref_data is not None and len(ref_data) > 0:
                np.save(str(folder / "ref_poses.npy"), ref_data)
                print(f"  ref_poses.npy  shape={ref_data.shape}")
                ref_saved = True
            else:
                print("  Warning: GT loader returned no data.", file=sys.stderr)
        else:
            print(
                f"  No GT loader for '{Path(executable).stem}' — "
                "use --ref-poses to provide GT manually."
            )

    if not ref_saved:
        print("  ref_poses.npy not saved; evaluation will run without GT.")

    print(f"\nDone. Sandbox at:\n  {folder}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

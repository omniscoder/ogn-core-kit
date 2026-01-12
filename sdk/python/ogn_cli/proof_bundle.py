from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import resolve_paths
from .util import print_json


MANIFEST_SCHEMA = "ogn_proof_bundle_manifest@1"


@dataclass(frozen=True)
class BundlePaths:
  run_dir: Path
  bundle_dir: Path


def _utc_now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
  h = hashlib.sha256()
  h.update(data)
  return h.hexdigest()


def _sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as handle:
    while True:
      chunk = handle.read(1024 * 1024)
      if not chunk:
        break
      h.update(chunk)
  return h.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
  if not path.is_file():
    return {}
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return {}


def _find_latest_run(runs_root: Path) -> Optional[Path]:
  if not runs_root.is_dir():
    return None
  candidates: List[Path] = []
  for profile_dir in runs_root.iterdir():
    if not profile_dir.is_dir():
      continue
    for run_dir in profile_dir.iterdir():
      if run_dir.is_dir():
        candidates.append(run_dir)
  if not candidates:
    return None
  return max(candidates)


def _resolve_run_dir(run_id: str, runs_root: Path) -> Path:
  token = (run_id or "").strip()
  if not token:
    raise RuntimeError("run_id is empty")

  if token.lower() in ("latest", "@latest"):
    latest = _find_latest_run(runs_root)
    if latest is None:
      raise RuntimeError(f"No runs found under {runs_root}")
    return latest

  candidate = Path(token).expanduser()
  if not candidate.is_absolute():
    rel = runs_root / candidate
    if rel.is_dir():
      return rel
    if candidate.is_dir():
      return candidate.resolve()
  else:
    if candidate.is_dir():
      return candidate.resolve()

  # Allow passing a bare timestamp directory name (e.g. 20251216T123000Z).
  matches: List[Path] = []
  if runs_root.is_dir():
    for profile_dir in runs_root.iterdir():
      if not profile_dir.is_dir():
        continue
      probe = profile_dir / token
      if probe.is_dir():
        matches.append(probe)
  if len(matches) == 1:
    return matches[0]
  if len(matches) > 1:
    rendered = "\n".join(f"  - {m}" for m in sorted(matches))
    raise RuntimeError(f"run_id {token!r} matched multiple runs:\n{rendered}")

  raise RuntimeError(
      f"Run not found for run_id={token!r}. Provide a run dir path, "
      f"'profile_sample/<timestamp>', a bare <timestamp>, or 'latest'."
  )


def _default_bundle_dir(run_dir: Path, runs_root: Path) -> Path:
  try:
    rel = run_dir.relative_to(runs_root)
  except ValueError:
    return Path.cwd() / "proof_bundles" / run_dir.name
  return runs_root / "proof_bundles" / rel


def _render_rel(path: Path, root: Path) -> str:
  try:
    return path.relative_to(root).as_posix()
  except Exception:
    return str(path)


def _copy_required(run_dir: Path, bundle_dir: Path) -> List[Path]:
  required = ["status.json", "metrics.json"]
  copied: List[Path] = []
  for name in required:
    src = run_dir / name
    if not src.is_file():
      raise RuntimeError(f"Missing required run artifact: {src}")
    dst = bundle_dir / name
    shutil.copyfile(src, dst)
    copied.append(dst)
  return copied


def _copy_optional(run_dir: Path, bundle_dir: Path) -> List[Path]:
  copied: List[Path] = []
  for name in ("config.json",):
    src = run_dir / name
    if not src.is_file():
      continue
    dst = bundle_dir / name
    shutil.copyfile(src, dst)
    copied.append(dst)
  return copied


def _format_seconds(val: Any) -> str:
  try:
    f = float(val)
  except Exception:
    return "unknown"
  if f < 0:
    return "unknown"
  if f >= 3600:
    return f"{f / 3600.0:.2f}h"
  if f >= 60:
    return f"{f / 60.0:.2f}m"
  return f"{f:.2f}s"


def _generate_summary_md(
    bundle_dir: Path,
    run_dir: Path,
    runs_root: Path,
    status: Dict[str, Any],
    metrics: Dict[str, Any],
) -> str:
  profile = status.get("profile") or metrics.get("profile") or "unknown"
  sample = status.get("sample") or metrics.get("sample") or "unknown"
  state = status.get("state") or "unknown"
  started = metrics.get("started_utc") or "unknown"
  finished = metrics.get("finished_utc") or "unknown"
  wall = _format_seconds(metrics.get("wall_time_seconds"))
  runtime = _format_seconds(metrics.get("runtime_seconds"))
  gpu = metrics.get("gpu") or {}
  gpu_name = gpu.get("device") or "none"
  gpu_count = gpu.get("count")
  try:
    gpu_count_str = str(int(gpu_count)) if gpu_count is not None else "unknown"
  except Exception:
    gpu_count_str = "unknown"

  vcf_path = metrics.get("vcf_path") or None
  validation_status = metrics.get("validation_status") or "unknown"
  accuracy = metrics.get("accuracy") or {}

  lines: List[str] = []
  lines.append("# OGN proof bundle")
  lines.append("")
  lines.append(f"- generated_utc: {_utc_now()}")
  lines.append(f"- run_id: {_render_rel(run_dir, runs_root)}")
  lines.append(f"- profile: {profile}")
  lines.append(f"- sample: {sample}")
  lines.append(f"- state: {state}")
  lines.append(f"- started_utc: {started}")
  lines.append(f"- finished_utc: {finished}")
  lines.append(f"- wall_time: {wall}")
  lines.append(f"- runtime: {runtime}")
  lines.append(f"- gpu: {gpu_name} (count={gpu_count_str})")
  if vcf_path:
    lines.append(f"- vcf_path: {vcf_path}")
  lines.append(f"- validation_status: {validation_status}")

  if isinstance(accuracy, dict) and accuracy:
    lines.append("")
    lines.append("## Accuracy (if present)")
    panel = accuracy.get("panel") or "unknown"
    status_str = (accuracy.get("status") or "unknown").upper()
    lines.append(f"- panel: {panel}")
    lines.append(f"- status: {status_str}")
    if accuracy.get("snp_f1") is not None:
      try:
        lines.append(f"- snp_f1: {float(accuracy.get('snp_f1')):.4f}")
      except Exception:
        lines.append(f"- snp_f1: {accuracy.get('snp_f1')}")
    if accuracy.get("indel_f1") is not None:
      try:
        lines.append(f"- indel_f1: {float(accuracy.get('indel_f1')):.4f}")
      except Exception:
        lines.append(f"- indel_f1: {accuracy.get('indel_f1')}")

  lines.append("")
  lines.append("## Bundle contents")
  lines.append("- status.json")
  lines.append("- metrics.json")
  if (bundle_dir / "config.json").is_file():
    lines.append("- config.json")
  lines.append("- summary.md")
  if (bundle_dir / "plots").is_dir():
    plots = sorted(p.name for p in (bundle_dir / "plots").iterdir() if p.is_file())
    if plots:
      lines.append("- plots/")
      for name in plots:
        lines.append(f"  - {name}")
  lines.append("- manifest.json")
  lines.append("- manifest.sha256")
  lines.append("- manifest.sig (optional)")
  lines.append("")
  lines.append("To verify integrity, check `manifest.sha256` (and `manifest.sig` if present).")
  return "\n".join(lines)


def _maybe_write_plots(bundle_dir: Path, metrics: Dict[str, Any]) -> List[Path]:
  """Best-effort PNG plots.

  We intentionally avoid heavy plotting deps in the core CLI. If Pillow is
  available, render small bar charts. Otherwise, skip plots cleanly.
  """

  try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
  except Exception:
    return []

  def render_bars(
      out_path: Path,
      *,
      title: str,
      ylabel: str,
      labels: List[str],
      values: List[float],
      y_max: float,
  ) -> None:
    width = 800
    height = 420
    margin_left = 90
    margin_right = 30
    margin_top = 60
    margin_bottom = 80

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Title
    draw.text((margin_left, 20), title, fill=(0, 0, 0), font=font)

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # Axes
    x0 = margin_left
    y0 = height - margin_bottom
    x1 = width - margin_right
    y1 = margin_top
    draw.line([(x0, y0), (x1, y0)], fill=(0, 0, 0))
    draw.line([(x0, y0), (x0, y1)], fill=(0, 0, 0))

    # Y label
    draw.text((10, y1), ylabel, fill=(0, 0, 0), font=font)

    n = max(1, len(values))
    bar_gap = max(8, int(plot_w * 0.04))
    total_gap = bar_gap * (n + 1)
    bar_w = max(12, int((plot_w - total_gap) / n))

    palette = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]

    for i, (label, value) in enumerate(zip(labels, values)):
      v = max(0.0, min(float(value), y_max))
      h_px = int((v / y_max) * plot_h) if y_max > 0 else 0
      left = x0 + bar_gap + i * (bar_w + bar_gap)
      right = left + bar_w
      top = y0 - h_px
      color = palette[i % len(palette)]
      draw.rectangle([(left, top), (right, y0)], fill=color, outline=(0, 0, 0))

      # Value label
      val_text = f"{v:.4f}" if y_max <= 1.0 else f"{v:.2f}"
      draw.text((left, top - 14), val_text, fill=(0, 0, 0), font=font)

      # X label
      draw.text((left, y0 + 10), label, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")

  plots_dir = bundle_dir / "plots"
  created: List[Path] = []

  # Accuracy plot (F1 bars)
  accuracy = metrics.get("accuracy") or {}
  if isinstance(accuracy, dict):
    labels: List[str] = []
    values: List[float] = []
    for key, label in (("snp_f1", "SNP F1"), ("indel_f1", "INDEL F1")):
      if accuracy.get(key) is None:
        continue
      try:
        labels.append(label)
        values.append(float(accuracy.get(key)))
      except Exception:
        continue
    if labels and values:
      out = plots_dir / "accuracy_f1.png"
      try:
        render_bars(out, title="Accuracy summary", ylabel="F1", labels=labels, values=values, y_max=1.0)
        created.append(out)
      except Exception:
        pass

  # Runtime plot (seconds bars)
  labels = []
  values = []
  for key, label in (("wall_time_seconds", "wall"), ("runtime_seconds", "runtime")):
    if metrics.get(key) is None:
      continue
    try:
      labels.append(label)
      values.append(max(0.0, float(metrics.get(key))))
    except Exception:
      continue
  if labels and values:
    out = plots_dir / "run_time.png"
    y_max = max(values) * 1.05 if max(values) > 0 else 1.0
    try:
      render_bars(out, title="Run time", ylabel="seconds", labels=labels, values=values, y_max=y_max)
      created.append(out)
    except Exception:
      pass

  return created


def _write_manifest(
    bundle_dir: Path, run_dir: Path, runs_root: Path, files: List[Path]
) -> Tuple[Path, str]:
  entries = []
  for path in sorted(files, key=lambda p: p.as_posix()):
    rel = path.relative_to(bundle_dir).as_posix()
    entries.append(
        {
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    )

  status = _read_json(bundle_dir / "status.json")
  metrics = _read_json(bundle_dir / "metrics.json")
  profile = status.get("profile") or metrics.get("profile") or "unknown"
  sample = status.get("sample") or metrics.get("sample") or "unknown"

  payload: Dict[str, Any] = {
      "schema": MANIFEST_SCHEMA,
      "generated_utc": _utc_now(),
      "run_dir": str(run_dir),
      "run_id": _render_rel(run_dir, runs_root),
      "profile": profile,
      "sample": sample,
      "files": entries,
  }

  manifest_path = bundle_dir / "manifest.json"
  manifest_path.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  manifest_hash = _sha256_bytes(manifest_path.read_bytes())
  (bundle_dir / "manifest.sha256").write_text(
      f"{manifest_hash}  manifest.json\n", encoding="utf-8"
  )
  return manifest_path, manifest_hash


def _maybe_sign_manifest(bundle_dir: Path, manifest_path: Path, sign_key: Optional[Path]) -> Optional[Path]:
  key = sign_key
  if key is None:
    env = os.environ.get("OGN_SIGNING_KEY")
    if env:
      key = Path(env).expanduser()
  if key is None or not key.is_file():
    return None

  openssl = shutil.which("openssl")
  if not openssl:
    return None

  try:
    sig_bytes = subprocess.check_output(
        [openssl, "dgst", "-sha256", "-sign", str(key), str(manifest_path)],
        stderr=subprocess.DEVNULL,
    )
  except Exception:
    return None

  sig_b64 = base64.b64encode(sig_bytes).decode("ascii")
  out = bundle_dir / "manifest.sig"
  out.write_text(sig_b64 + "\n", encoding="utf-8")
  return out


def build_proof_bundle(
    *,
    run_id: str,
    runs_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    overwrite: bool = False,
    no_plots: bool = False,
    sign_key: Optional[Path] = None,
) -> BundlePaths:
  paths = resolve_paths(runs_root=str(runs_root) if runs_root else None)
  run_dir = _resolve_run_dir(run_id, paths.runs_root)

  bundle_dir = out_dir or _default_bundle_dir(run_dir, paths.runs_root)
  bundle_dir = bundle_dir.expanduser().resolve()
  if bundle_dir.exists():
    if not overwrite:
      raise RuntimeError(f"Bundle output already exists: {bundle_dir}")
    shutil.rmtree(bundle_dir)
  bundle_dir.mkdir(parents=True, exist_ok=True)

  copied = []
  copied.extend(_copy_required(run_dir, bundle_dir))
  copied.extend(_copy_optional(run_dir, bundle_dir))

  status = _read_json(bundle_dir / "status.json")
  metrics = _read_json(bundle_dir / "metrics.json")

  summary_path = bundle_dir / "summary.md"
  plot_files: List[Path] = []
  if not no_plots:
    plot_files = _maybe_write_plots(bundle_dir, metrics)

  summary_path.write_text(
      _generate_summary_md(bundle_dir, run_dir, paths.runs_root, status, metrics) + "\n",
      encoding="utf-8",
  )

  manifest_inputs = copied + [summary_path] + plot_files
  manifest_path, _ = _write_manifest(bundle_dir, run_dir, paths.runs_root, manifest_inputs)
  sig_path = _maybe_sign_manifest(bundle_dir, manifest_path, sign_key)
  if sig_path:
    # Signature is not listed in manifest.json; it is tied to the manifest hash.
    pass

  return BundlePaths(run_dir=run_dir, bundle_dir=bundle_dir)


def add_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser(
      "proof-bundle",
      help="Generate a shareable proof bundle for a run (status/metrics/summary + manifest).",
  )
  parser.add_argument(
      "run_id",
      metavar="RUN_ID",
      help="Run identifier: run dir path, 'profile_sample/<timestamp>', bare <timestamp>, or 'latest'.",
  )
  parser.add_argument(
      "--runs-root",
      metavar="PATH",
      help="Override runs root (default: $OGN_RUNS_ROOT or ./runs).",
  )
  parser.add_argument(
      "--out",
      metavar="PATH",
      help="Output directory for the proof bundle (default: <runs_root>/proof_bundles/<run_rel>/).",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Overwrite bundle output directory if it already exists.",
  )
  parser.add_argument(
      "--no-plots",
      action="store_true",
      help="Do not generate optional PNG plots (default: generate when Pillow is available).",
  )
  parser.add_argument(
      "--sign-key",
      metavar="PATH",
      help="PEM private key used to sign manifest.json (default: $OGN_SIGNING_KEY).",
  )
  parser.add_argument(
      "--json",
      action="store_true",
      help="Emit a JSON summary (bundle_dir/run_dir) instead of human text.",
  )


def run_proof_bundle(args: argparse.Namespace) -> int:
  try:
    out_dir = Path(args.out).expanduser() if args.out else None
    runs_root = Path(args.runs_root).expanduser() if args.runs_root else None
    sign_key = Path(args.sign_key).expanduser() if args.sign_key else None
    bundle = build_proof_bundle(
        run_id=args.run_id,
        runs_root=runs_root,
        out_dir=out_dir,
        overwrite=bool(args.overwrite),
        no_plots=bool(args.no_plots),
        sign_key=sign_key,
    )
  except Exception as ex:
    print(f"Error: {ex}", file=sys.stderr)
    return 2

  summary = {"run_dir": str(bundle.run_dir), "bundle_dir": str(bundle.bundle_dir)}
  if getattr(args, "json", False):
    print_json(summary)
  else:
    print(f"[proof-bundle] run_dir={summary['run_dir']}")
    print(f"[proof-bundle] bundle_dir={summary['bundle_dir']}")
  return 0


def main(argv: Optional[List[str]] = None) -> int:
  parser = argparse.ArgumentParser(prog="ogn-proof-bundle", description="OGN proof bundle generator")
  parser.add_argument("run_id", help="Run identifier (path, timestamp, or 'latest').")
  parser.add_argument("--runs-root", help="Override runs root (default: $OGN_RUNS_ROOT or ./runs).")
  parser.add_argument("--out", help="Output directory for the proof bundle.")
  parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")
  parser.add_argument("--no-plots", action="store_true", help="Disable optional PNG plots.")
  parser.add_argument("--sign-key", help="PEM private key to sign manifest.json (default: $OGN_SIGNING_KEY).")
  parser.add_argument("--json", action="store_true", help="Emit a JSON summary.")
  args = parser.parse_args(sys.argv[1:] if argv is None else argv)
  return run_proof_bundle(args)

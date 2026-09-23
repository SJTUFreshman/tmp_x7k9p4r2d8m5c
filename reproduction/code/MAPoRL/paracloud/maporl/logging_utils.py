"""Run artifacts: manifest, per-iteration metrics, atomic state."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def vendor_hashes(vendor_dir: Path) -> dict[str, str]:
    return {
        p.name: sha256_file(p)
        for p in sorted(Path(vendor_dir).glob("*"))
        if p.is_file() and p.suffix in {".py", ".sh"}
    }


def build_manifest(
    config: Any,
    *,
    root: Path,
    dataset_files: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to say what produced a number."""
    manifest: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config.to_dict(),
        "config_sha256": config.sha256(),
        "vendor_sha256": vendor_hashes(Path(root) / "vendor"),
        "reward_mode": getattr(config, "reward_mode", None),
        "joint_mode": "synchronous_debate",
        "gpu_plan": config.serving.gpu_plan,
        "lora_naming": config.serving.lora_naming,
        "prefix_caching": config.serving.enable_prefix_caching,
        "hyperparameter_origin": (
            "MAPoRL source-derived defaults with documented adaptations for "
            "three heterogeneous agents and five datasets; see PAPER_NOTES.md."
        ),
        "guided_decoding": False,
        "logprobs_mode": "processed_logprobs",
        "policy_layout": "turn 0 frozen base; later turns shared collaboration LoRA",
        "checkpoint_scope": "policy, critics, optimizers, prompt pool, RNG states",
    }
    if dataset_files:
        manifest["dataset_files"] = dataset_files
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "vllm", "sympy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest["package_versions"] = versions
    if extra:
        manifest.update(extra)
    return manifest


class RunLogger:
    """One run directory: metrics.jsonl, state.json, manifest.json."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.state_path = self.run_dir / "state.json"
        self.manifest_path = self.run_dir / "manifest.json"

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        write_json_atomic(self.manifest_path, manifest)

    def log_metrics(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.metrics_path, payload)

    def log_invocation(self, payload: dict[str, Any]) -> None:
        """Record run-time overrides without changing the committed config."""
        append_jsonl(self.run_dir / "logs" / "invocations.jsonl", {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hostname": socket.gethostname(),
            "process_id": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            **payload,
        })

    def save_state(self, payload: dict[str, Any]) -> None:
        write_json_atomic(self.state_path, payload)

    def load_state(self) -> dict[str, Any] | None:
        if not self.state_path.is_file():
            return None
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def print(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with (self.run_dir / "logs" / "loop.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

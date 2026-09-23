#!/usr/bin/env python3
"""Launch vLLM while making local dense-DP rendezvous ports rank-specific."""

from __future__ import annotations

import importlib
import multiprocessing
import os
import re
import runpy
import socket
import sys
from collections import defaultdict
from typing import Callable


_ENGINE_RANK = re.compile(r"EngineCore_DP(\d+)")
_COUNTERS: defaultdict[int, int] = defaultdict(int)


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("", port))
        except OSError:
            return False
    return True


def _rank_port(original: Callable[[], int]) -> int:
    process_name = multiprocessing.current_process().name
    match = _ENGINE_RANK.search(process_name)
    if match is None:
        if os.environ.get("CONIFER_DP_PORT_DEBUG") == "1":
            print(f"[deterministic-port] process={process_name} rank=none original", flush=True)
        return original()
    rank = int(match.group(1))
    base = int(os.environ.get("CONIFER_DP_PORT_BASE", "52000"))
    stride = int(os.environ.get("CONIFER_DP_PORT_STRIDE", "32"))
    if base < 1024 or stride < 4:
        raise RuntimeError("CONIFER_DP_PORT_BASE/STRIDE specify an invalid port range")
    first = base + rank * stride
    last = first + stride - 1
    offset = _COUNTERS[rank]
    _COUNTERS[rank] += 1
    for port in range(first + offset, last + 1):
        if _port_is_available(port):
            if os.environ.get("CONIFER_DP_PORT_DEBUG") == "1":
                print(f"[deterministic-port] process={process_name} rank={rank} port={port}", flush=True)
            return port
    for port in range(first, first + offset):
        if _port_is_available(port):
            return port
    raise RuntimeError(f"No free deterministic vLLM port for DP rank {rank}: {first}-{last}")


def _patch_vllm_port_allocators() -> None:
    modules = (
        "vllm.utils.network_utils",
        "vllm.v1.executor.multiproc_executor",
        "vllm.v1.executor.uniproc_executor",
        "vllm.v1.executor.ray_executor",
        "vllm.v1.utils",
        "vllm.distributed.device_communicators.shm_broadcast",
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common",
    )
    network_utils = importlib.import_module("vllm.utils.network_utils")
    original = network_utils.get_open_port
    patched = lambda: _rank_port(original)
    network_utils.get_open_port = patched
    for module_name in modules[1:]:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "get_open_port"):
            module.get_open_port = patched


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"cli", "api"}:
        raise SystemExit("usage: serve_vllm_deterministic.py {cli|api} [vLLM arguments]")
    entrypoint = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    if entrypoint == "cli":
        module = importlib.import_module("vllm.entrypoints.cli.main")
        module.main()
    else:
        runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


_patch_vllm_port_allocators()

if __name__ == "__main__":
    main()

r"""GPTSwarm DAG definition + async topological execution.

The swarm is a fixed DAG of Node instances. Execution runs nodes in
topological layers, with concurrent execution within a layer via
asyncio.gather(). The final answer is the answer of the sink node
(convention: named "output" or the unique node with no outgoing edges).

Fixed swarm structure (per-problem, 7 executor nodes):

                       [ INPUT ]
                    /     |      \
                   /      |       \
                 IO-0    CoT-0    CoT-1        (layer 1: 3 nodes, parallel)
                    \     |      /
                     \    |     /
                     [ Debate-0 ]              (layer 2: 1 node)
                     /    |     \
                    /     |      \
                 CoT-2    |      IO-1          (layer 3: 2 nodes, parallel)
                    \     |     /
                     \    |    /
                    [ Aggregator ]             (layer 4: 1 node = final answer)
                          |
                       [ OUTPUT ]

Layer 4 (Aggregator) consumes ALL of {Debate-0, CoT-2, IO-1}. The DAG
is deliberately not a chain — Aggregator sees both the debated answer
AND the two fresh second-layer answers.

Layer 2 (Debate-0) consumes ALL of layer 1: {IO-0, CoT-0, CoT-1}.
Layer 3 nodes each consume ONLY Debate-0 (single-input second-layer
reasoning). This gives width at layers 1 & 3 and a synthesis
bottleneck at Debate-0 and Aggregator.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from nodes import Node, NodeOutput, make_node


LLMCaller = Callable[[List[Dict[str, str]]], str]
CallerRouter = Mapping[str, LLMCaller]


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------


@dataclass
class SwarmDAG:
    """A fixed DAG: nodes + directed edges (parent -> child)."""

    nodes: Dict[str, Node]                              # name -> Node
    edges: List[Tuple[str, str]]                        # (parent, child)
    sink_name: str                                      # final node
    _incoming: Dict[str, List[str]] = field(default_factory=dict)
    _outgoing: Dict[str, List[str]] = field(default_factory=dict)
    _layers: List[List[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in self.nodes:
            self._incoming[name] = []
            self._outgoing[name] = []
        for parent, child in self.edges:
            if parent not in self.nodes:
                raise ValueError(f"edge references unknown parent {parent!r}")
            if child not in self.nodes:
                raise ValueError(f"edge references unknown child {child!r}")
            self._outgoing[parent].append(child)
            self._incoming[child].append(parent)
        if self.sink_name not in self.nodes:
            raise ValueError(f"sink {self.sink_name!r} not in nodes")
        self._layers = _kahn_layers(self.nodes.keys(), self._incoming)


def _kahn_layers(
    names: Any,
    incoming: Dict[str, List[str]],
) -> List[List[str]]:
    """Layered topological sort (Kahn's algorithm with layer batches)."""
    remaining_in = {n: list(incoming.get(n, [])) for n in names}
    layers: List[List[str]] = []
    while remaining_in:
        current_layer = [n for n, ins in remaining_in.items() if not ins]
        if not current_layer:
            raise ValueError("Cycle detected in swarm DAG")
        current_layer.sort()
        layers.append(current_layer)
        for n in current_layer:
            del remaining_in[n]
        for n in remaining_in:
            remaining_in[n] = [p for p in remaining_in[n] if p in remaining_in]
    return layers


# ---------------------------------------------------------------------------
# Fixed swarm factory
# ---------------------------------------------------------------------------


def build_fixed_swarm() -> SwarmDAG:
    """Return the fixed 7-node swarm described in the module docstring."""
    node_specs: List[Tuple[str, str]] = [
        # (name, node_type)
        ("io_0", "IO"),
        ("cot_0", "CoT"),
        ("cot_1", "CoT"),
        ("debate_0", "Debate"),
        ("cot_2", "CoT"),
        ("io_1", "IO"),
        ("aggregator", "Aggregator"),
    ]
    nodes = {name: make_node(node_type, name) for name, node_type in node_specs}
    edges: List[Tuple[str, str]] = [
        # Layer 1 -> Layer 2 (Debate consumes all of IO-0, CoT-0, CoT-1)
        ("io_0", "debate_0"),
        ("cot_0", "debate_0"),
        ("cot_1", "debate_0"),
        # Layer 2 -> Layer 3 (fresh reasoning off Debate-0)
        ("debate_0", "cot_2"),
        ("debate_0", "io_1"),
        # Layer 2 & Layer 3 -> Layer 4 (Aggregator sees Debate-0, CoT-2, IO-1)
        ("debate_0", "aggregator"),
        ("cot_2", "aggregator"),
        ("io_1", "aggregator"),
    ]
    return SwarmDAG(nodes=nodes, edges=edges, sink_name="aggregator")


# ---------------------------------------------------------------------------
# Execution record
# ---------------------------------------------------------------------------


@dataclass
class SwarmRunRecord:
    problem_id: str
    node_outputs: List[NodeOutput] = field(default_factory=list)
    layers: List[List[str]] = field(default_factory=list)
    final_answer: Optional[str] = None
    error: Optional[str] = None
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


async def _run_swarm_async(
    dag: SwarmDAG,
    problem: Any,
    callers: CallerRouter,
) -> SwarmRunRecord:
    record = SwarmRunRecord(
        problem_id=getattr(problem, "id", ""),
        layers=[list(layer) for layer in dag._layers],
    )
    outputs: Dict[str, NodeOutput] = {}
    started = time.monotonic()

    try:
        for layer in dag._layers:
            tasks = []
            for name in layer:
                node = dag.nodes[name]
                preds = [outputs[p] for p in dag._incoming[name]]
                # A per-node route overrides the legacy node-type route. This
                # lets experiments rotate models without changing the DAG.
                caller = callers.get(name)
                if caller is None:
                    caller = callers.get(node.node_type)
                if caller is None:
                    raise ValueError(
                        f"No caller configured for node {name!r} "
                        f"(type {node.node_type!r})"
                    )
                tasks.append(node.run(problem, preds, caller))
            results = await asyncio.gather(*tasks)
            for name, out in zip(layer, results):
                outputs[name] = out
                record.node_outputs.append(out)

        sink_output = outputs.get(dag.sink_name)
        if sink_output is None:
            record.error = f"Sink node {dag.sink_name!r} produced no output"
        else:
            record.final_answer = sink_output.answer
    except Exception as exc:
        record.error = f"{type(exc).__name__}: {exc}"

    record.wall_time_s = round(time.monotonic() - started, 3)
    return record


def run_swarm_on_problem(
    dag: SwarmDAG,
    problem: Any,
    callers: CallerRouter,
) -> SwarmRunRecord:
    """Sync entry point: runs the swarm on one MuSiQue problem."""
    return asyncio.run(_run_swarm_async(dag, problem, callers))


# ---------------------------------------------------------------------------
# Problem adapter
# ---------------------------------------------------------------------------


@dataclass
class SwarmProblem:
    """View passed into nodes. Hides raw MuSiQueProblem internals."""

    id: str
    question: str
    rendered_text: str

    @classmethod
    def from_musique(cls, problem: Any) -> "SwarmProblem":
        from jca.src.data import format_problem_as_prompt
        return cls(
            id=problem.id,
            question=problem.question,
            rendered_text=format_problem_as_prompt(problem),
        )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def swarm_record_to_dict(record: SwarmRunRecord) -> Dict[str, Any]:
    return {
        "problem_id": record.problem_id,
        "layers": record.layers,
        "node_outputs": [asdict(o) for o in record.node_outputs],
        "final_answer": record.final_answer,
        "error": record.error,
        "wall_time_s": record.wall_time_s,
    }

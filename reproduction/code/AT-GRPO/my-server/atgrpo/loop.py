"""The AT-GRPO online iteration loop.

One iteration:

  1. draw B prompts
  2. tree-structured rollout: branch K ways at each turn, continue each to the
     end so a candidate is scored by its consequences
  3. reward every candidate by its episode outcome   [one batched grader call]
  4. normalize within each (agent, turn) group -> one advantage per candidate
  5. partition candidates by acting agent; one optimizer step per agent
  6. hot-swap the new adapters into the servers
  7. commit metrics and state

Differences from the MAGRPO loop in this repo, both of which matter:

* the rollout branches per turn instead of sampling independent episodes, so
  every group has K members sharing one prompt rather than one member;
* the advantage is normalized per (agent, turn) rather than broadcast from a
  single team reward.

``commit`` is the last step, so a crash mid-iteration is replayed rather than
half-applied.
"""
from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .at_advantage import rows_by_agent, summarize
from .config import AGENTS, Config
from .logging_utils import RunLogger, project_code_sha256
from .pool import PromptPool, expected_prescreen_count, load_allowlist
from .tasks.base import Problem, TaskAdapter
from .tree_rollout import run_tree, score_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_agent_updates(active_agents, trainers, update_agent):
    """Run independent agent updates together only when devices are disjoint."""
    devices = [str(trainers[agent].device) for agent in active_agents]
    can_parallelize = (
        len(active_agents) > 1
        and len(set(devices)) == len(devices)
        and all(device.startswith("cuda") for device in devices)
    )
    if not can_parallelize:
        return [update_agent(agent) for agent in active_agents]
    with ThreadPoolExecutor(max_workers=len(active_agents)) as pool:
        return list(pool.map(update_agent, active_agents))


@dataclass
class IterationResult:
    iteration: int
    n_groups: int
    n_degenerate: int
    reward_mean: float
    agent_stats: dict[str, Any]
    seconds: float


class ATGRPOLoop:
    def __init__(
        self,
        config: Config,
        task: TaskAdapter,
        fleet: Any,
        trainers: dict[str, Any],
        *,
        run_dir: Path,
        callers_factory: Callable[[dict[str, str]], dict[str, Any]],
        prescreen_path: Path | None = None,
        code_sha256: str | None = None,
    ) -> None:
        self.config = config
        self.task = task
        self.fleet = fleet
        self.trainers = trainers
        self.run_dir = Path(run_dir)
        self.callers_factory = callers_factory
        self.logger = RunLogger(self.run_dir)
        self.adapters_dir = self.run_dir / "adapters"
        self.traj_dir = self.run_dir / "rollouts"
        self.code_sha256 = code_sha256 or project_code_sha256(PROJECT_ROOT)

        problems = task.load("train")
        if config.pool.prescreen:
            if prescreen_path is None:
                raise ValueError(
                    "pool.prescreen is enabled but no prescreen allowlist was provided"
                )
            source_ids = {problem.problem_id for problem in problems}
            allow = load_allowlist(
                prescreen_path,
                expected_group_size=config.atgrpo.group_size_K,
                expected_count=expected_prescreen_count(
                    config.pool.prescreen_sample, len(problems)
                ),
                expected_keep_band=config.pool.keep_band,
                source_ids=source_ids,
            )
            if allow is None:
                raise ValueError(f"prescreen allowlist not found: {prescreen_path}")
        else:
            allow = None
        self.pool = PromptPool(problems, seed=config.pool.shuffle_seed, allowlist=allow)
        self.logger.print(
            f"pool: {len(self.pool.problems)} prompts"
            + (f" (prescreened from {len(problems)})" if allow is not None else "")
        )
        self.iteration = 0
        self.degenerate_streak = 0
        self.started_at = time.time()
        self.best: dict[str, Any] = {}

    # -- one iteration ------------------------------------------------------
    def run_iteration(self) -> IterationResult:
        started = time.time()
        cfg = self.config
        at = cfg.atgrpo
        iteration = self.iteration

        callers = self.callers_factory(self.fleet.base_urls())
        batch = self.pool.next_batch(at.prompts_per_iter)

        t_rollout = time.time()
        rollouts = run_tree(
            self.task, batch, callers,
            agents=AGENTS[: at.agent_num],
            round_num=at.round_num,
            group_size=at.group_size_K,
            seed=cfg.seed,
            iteration=iteration,
            branch_mode=at.branch_mode,
            step_retries=cfg.rollout.step_retries,
            max_workers=cfg.rollout.max_concurrency,
        )
        rollout_s = time.time() - t_rollout

        failed_rollouts = [rollout for rollout in rollouts if rollout.error]
        if failed_rollouts:
            examples = "; ".join(
                f"{rollout.problem_id}: {rollout.error}"
                for rollout in failed_rollouts[:3]
            )
            raise RuntimeError(
                f"rollout infrastructure failed for {len(failed_rollouts)}/"
                f"{len(rollouts)} prompts; refusing to score, train, or commit "
                f"a partial iteration. Examples: {examples}"
            )

        t_reward = time.time()
        score_candidates(self.task, list(zip(batch, rollouts)))
        reward_s = time.time() - t_reward

        kept, report = summarize(
            rollouts,
            clip=at.clip_advantage,
            std_floor=at.std_floor,
            strict_fingerprint=at.strict_fingerprint,
        )
        self._check_alarms(report)

        rows = rows_by_agent(kept, AGENTS[: at.agent_num])

        t_train = time.time()
        active_agents = AGENTS[: at.agent_num]

        def update_agent(agent: str) -> tuple[str, dict[str, Any], Path]:
            trainer = self.trainers[agent]
            stats = trainer.step(
                rows[agent],
                iteration=iteration,
                clip_epsilon=at.clip_epsilon,
                kl_coef=at.kl_coef,
                loss_agg=at.loss_agg,
                inner_epochs=at.inner_epochs,
                max_encode_skip_frac=cfg.train.max_encode_skip_frac,
                ratio_tolerance=cfg.train.ratio_tolerance,
            )
            # Republish even on a skipped step so the swap stays uniform.
            adapter_path = trainer.save_adapter(
                self.adapters_dir / agent / f"iter_{iteration:04d}"
            )
            return agent, stats.to_dict(), adapter_path

        updates = run_agent_updates(active_agents, self.trainers, update_agent)

        agent_stats = {agent: stats for agent, stats, _ in updates}
        published = {agent: path for agent, _, path in updates}
        train_s = time.time() - t_train

        t_swap = time.time()
        self.fleet.publish_adapters(iteration, published)
        swap_s = time.time() - t_swap

        self._write_rollouts(iteration, rollouts)

        total_s = time.time() - started
        metrics = {
            "iteration": iteration,
            "n_prompts": len(batch),
            "n_groups": report.n_groups,
            "n_degenerate": report.n_degenerate,
            "degenerate_frac": report.degenerate_frac,
            "n_candidates": report.n_candidates,
            "n_trainable": report.n_assigned,
            # Must stay 0: a non-zero value means a group compared candidates
            # that were not sampled from the same state.
            "fingerprint_violations": report.n_fingerprint_violations,
            "reward": {
                "mean": report.reward_mean,
                "std": report.reward_std,
                "min": report.reward_min,
                "max": report.reward_max,
            },
            "advantage_abs_mean": report.advantage_abs_mean,
            "rows_per_agent": dict(report.per_agent),
            "rows_per_turn": {str(k): v for k, v in sorted(report.per_turn.items())},
            "protocol_error_rate": self._protocol_error_rate(rollouts),
            "agents": agent_stats,
            "timing": {
                "rollout_s": rollout_s,
                "reward_s": reward_s,
                "train_s": train_s,
                "swap_s": swap_s,
                "total_s": total_s,
            },
        }
        self.logger.log_metrics(metrics)
        self.iteration += 1
        self.save_state()

        rows_summary = " ".join(
            f"{a}:{agent_stats[a]['n_rows']}" for a in AGENTS[: at.agent_num]
        )
        self.logger.print(
            f"iter {iteration}: reward={report.reward_mean:.4f} "
            f"degenerate={report.n_degenerate}/{report.n_groups} "
            f"rows=[{rows_summary}] {total_s:.1f}s"
        )
        return IterationResult(
            iteration=iteration,
            n_groups=report.n_groups,
            n_degenerate=report.n_degenerate,
            reward_mean=report.reward_mean,
            agent_stats=agent_stats,
            seconds=total_s,
        )

    # -- alarms -------------------------------------------------------------
    def _check_alarms(self, report) -> None:
        at = self.config.atgrpo
        if report.n_fingerprint_violations:
            raise RuntimeError(
                f"{report.n_fingerprint_violations} groups had divergent prompts. "
                "AT-GRPO's advantage is only meaningful within a frozen state; "
                "this means the tree sampler leaked history into a group."
            )
        if report.n_groups and report.degenerate_frac > at.max_degenerate_frac:
            self.degenerate_streak += 1
        else:
            self.degenerate_streak = 0
        if self.degenerate_streak >= at.degenerate_patience:
            raise RuntimeError(
                f"{self.degenerate_streak} consecutive iterations with "
                f">{at.max_degenerate_frac:.0%} degenerate (agent, turn) groups: "
                "every candidate in a group is scoring the same, so no gradient "
                "is produced. Raise group_size_K, or prescreen the pool."
            )

    @staticmethod
    def _protocol_error_rate(rollouts) -> float:
        total = errors = 0
        for rollout in rollouts:
            for candidate in rollout.candidates:
                total += 1
                errors += 1 if candidate.protocol_error else 0
        return errors / total if total else 0.0

    def _write_rollouts(self, iteration: int, rollouts) -> None:
        path = self.traj_dir / f"iter_{iteration:04d}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            for rollout in rollouts:
                handle.write(json.dumps(rollout.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(path)

    # -- resume -------------------------------------------------------------
    def save_state(self) -> None:
        self.logger.save_state(
            {
                "schema_version": 2,
                "iteration": self.iteration,
                "pool": self.pool.state(),
                "degenerate_streak": self.degenerate_streak,
                "elapsed_seconds": time.time() - self.started_at,
                "config_sha256": self.config.sha256(),
                "code_sha256": self.code_sha256,
                "best": self.best,
                "adapters": {
                    a: str(self.adapters_dir / a / f"iter_{self.iteration - 1:04d}")
                    for a in AGENTS[: self.config.atgrpo.agent_num]
                },
            }
        )

    def load_state(self) -> bool:
        state = self.logger.load_state()
        if not state:
            return False
        if state.get("config_sha256") != self.config.sha256():
            raise RuntimeError(
                "config changed since this run was created; resume would mix two "
                "different setups. Use a new run_id."
            )
        if state.get("code_sha256") != self.code_sha256:
            raise RuntimeError(
                "code changed since this run was created, or the checkpoint "
                "predates code fingerprinting; resume would mix training "
                "semantics. Use a new run_id."
            )
        self.iteration = int(state["iteration"])
        self.pool.load_state(state["pool"])
        self.degenerate_streak = int(state.get("degenerate_streak", 0))
        self.best = state.get("best") or {}
        self.logger.print(f"resumed at iteration {self.iteration}")
        return True

    # -- driver -------------------------------------------------------------
    def run(self, *, max_iterations: int | None = None) -> None:
        budget = self.config.budget
        limit = max_iterations or budget.max_iterations
        deadline = (
            self.started_at + budget.max_wall_clock_hours * 3600
            if budget.max_wall_clock_hours > 0
            else None
        )
        while self.iteration < limit:
            if deadline is not None and time.time() > deadline:
                self.logger.print("wall-clock budget reached; stopping")
                break
            self.run_iteration()
            self._prune_adapters()

    def _prune_adapters(self) -> None:
        import shutil

        budget = self.config.budget
        for agent in AGENTS[: self.config.atgrpo.agent_num]:
            directory = self.adapters_dir / agent
            if not directory.is_dir():
                continue
            checkpoints = sorted(
                (p for p in directory.iterdir() if p.is_dir() and p.name.startswith("iter_")),
                key=lambda p: p.name,
            )
            recent = set(checkpoints[-budget.keep_last :])
            keep = set(recent)
            for path in checkpoints:
                if int(path.name.split("_")[1]) % budget.save_every == 0:
                    keep.add(path)
            best_iter = (self.best or {}).get("iteration")
            if best_iter is not None:
                keep.add(directory / f"iter_{int(best_iter):04d}")
            for path in checkpoints:
                if path not in keep:
                    shutil.rmtree(path, ignore_errors=True)
                elif path not in recent:
                    (path / "optimizer.pt").unlink(missing_ok=True)

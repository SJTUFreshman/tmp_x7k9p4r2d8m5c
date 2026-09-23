"""The online MAGRPO iteration loop.

One iteration:

  1. draw B prompts
  2. sample G joint rollouts per prompt          [vLLM, current adapters]
  3. score each joint rollout with one team reward
  4. group-normalize -> one advantage per rollout, broadcast to all its turns
  5. partition turns by agent, one optimizer step per agent
  6. hot-swap the new adapters into the servers
  7. commit metrics and state

Steps 2 and 5 are the GPU-heavy ones and are strictly sequential, which is what
makes the static 4/4 GPU split viable: serving and training never share a device
and never run at the same time.

The loop is idempotent at iteration granularity -- ``commit`` is the last thing
that happens, so a crash mid-iteration is replayed rather than half-applied.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .advantage import summarize_groups
from .config import AGENTS, Config
from .logging_utils import RunLogger, build_manifest
from .pool import PromptPool, load_allowlist
from .rollout import run_group, score_groups
from .tasks.base import Problem, TaskAdapter
from .trajectory import JointTrajectory, write_trajectories


@dataclass
class IterationResult:
    iteration: int
    n_groups: int
    n_degenerate: int
    reward_mean: float
    agent_stats: dict[str, Any]
    seconds: float


class MagrpoLoop:
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
    ) -> None:
        self.config = config
        self.task = task
        self.fleet = fleet
        self.trainers = trainers
        self.run_dir = Path(run_dir)
        self.callers_factory = callers_factory
        self.logger = RunLogger(self.run_dir)
        self.adapters_dir = self.run_dir / "adapters"
        self.traj_dir = self.run_dir / "trajectories"

        problems = task.load("train")
        allow = load_allowlist(prescreen_path) if config.pool.prescreen else None
        self.pool = PromptPool(problems, seed=config.pool.shuffle_seed, allowlist=allow)
        self.logger.print(
            f"pool: {len(self.pool.problems)} prompts"
            + (f" (prescreened from {len(problems)})" if allow else "")
        )
        self.iteration = 0
        self.degenerate_streak = 0
        self.started_at = time.time()
        self.best: dict[str, Any] = {}

    # -- one iteration ------------------------------------------------------
    def run_iteration(self) -> IterationResult:
        started = time.time()
        cfg = self.config
        iteration = self.iteration

        callers = self.callers_factory(self.fleet.base_urls())
        batch = self.pool.next_batch(cfg.magrpo.prompts_per_iter)

        t_rollout = time.time()
        groups: list[tuple[Problem, list[JointTrajectory]]] = []
        for problem in batch:
            group = run_group(
                self.task,
                problem,
                callers,
                group_size=cfg.magrpo.group_size_G,
                iteration=iteration,
                t_max=cfg.rollout.t_max,
                seed=cfg.seed,
                joint_mode=cfg.rollout.joint_mode,
                step_retries=cfg.rollout.step_retries,
                shaping=cfg.magrpo.reward_shaping == "turn_level",
                max_workers=cfg.rollout.max_concurrency,
            )
            groups.append((problem, group))
        rollout_s = time.time() - t_rollout

        t_reward = time.time()
        score_groups(self.task, groups)
        reward_s = time.time() - t_reward

        kept, report = summarize_groups(
            [g for _, g in groups],
            granularity=cfg.magrpo.advantage_granularity,
            gamma=cfg.magrpo.gamma,
            clip=cfg.magrpo.clip_advantage,
            std_floor=cfg.magrpo.std_floor,
        )

        # Top up with extra prompts when too many groups were degenerate, so a
        # saturated pool does not silently starve the update.
        if cfg.magrpo.dynamic_sampling and report.n_groups:
            budget = int(cfg.magrpo.prompts_per_iter * cfg.magrpo.dynamic_sampling_max_factor)
            drawn = len(batch)
            while (
                report.n_groups - report.n_degenerate < cfg.magrpo.prompts_per_iter // 2
                and drawn < budget
            ):
                extra = self.pool.next_batch(cfg.magrpo.prompts_per_iter // 2)
                drawn += len(extra)
                new_groups = []
                for problem in extra:
                    new_groups.append(
                        (
                            problem,
                            run_group(
                                self.task, problem, callers,
                                group_size=cfg.magrpo.group_size_G,
                                iteration=iteration,
                                t_max=cfg.rollout.t_max,
                                seed=cfg.seed + drawn,
                                joint_mode=cfg.rollout.joint_mode,
                                step_retries=cfg.rollout.step_retries,
                                shaping=cfg.magrpo.reward_shaping == "turn_level",
                                max_workers=cfg.rollout.max_concurrency,
                            ),
                        )
                    )
                score_groups(self.task, new_groups)
                groups.extend(new_groups)
                kept, report = summarize_groups(
                    [g for _, g in groups],
                    granularity=cfg.magrpo.advantage_granularity,
                    gamma=cfg.magrpo.gamma,
                    clip=cfg.magrpo.clip_advantage,
                    std_floor=cfg.magrpo.std_floor,
                )

        self._check_degeneracy(report)

        rows_by_agent = {agent: [] for agent in AGENTS}
        for traj in kept:
            for turn in traj.turns:
                if turn.parsed is None or turn.advantage is None:
                    continue  # unparseable turns carry no usable target
                rows_by_agent[turn.agent].append(
                    {
                        "prompt_messages": turn.prompt_messages,
                        "response": turn.response,
                        "advantage": turn.advantage,
                    }
                )

        t_train = time.time()
        agent_stats: dict[str, Any] = {}
        published: dict[str, Path] = {}
        for agent in AGENTS:
            trainer = self.trainers[agent]
            stats = trainer.step(
                rows_by_agent[agent],
                iteration=iteration,
                clip_epsilon=cfg.magrpo.clip_epsilon,
                kl_coef=cfg.magrpo.kl_coef,
                loss_agg=cfg.magrpo.loss_agg,
                inner_epochs=cfg.magrpo.inner_epochs,
                max_encode_skip_frac=cfg.train.max_encode_skip_frac,
                ratio_tolerance=cfg.train.ratio_tolerance,
            )
            agent_stats[agent] = stats.to_dict()
            # Republish even on a skipped step so the swap stays uniform.
            published[agent] = trainer.save_adapter(
                self.adapters_dir / agent / f"iter_{iteration:04d}"
            )
        train_s = time.time() - t_train

        t_swap = time.time()
        self.fleet.publish_adapters(iteration, published)
        swap_s = time.time() - t_swap

        write_trajectories(
            self.traj_dir / f"iter_{iteration:04d}.jsonl.gz",
            [t for _, g in groups for t in g],
        )

        total_s = time.time() - started
        metrics = {
            "iteration": iteration,
            "n_prompts": len(groups),
            "n_groups": report.n_groups,
            "n_degenerate": report.n_degenerate,
            "degenerate_frac": report.degenerate_frac,
            "reward": {
                "mean": report.reward_mean,
                "std": report.reward_std,
                "min": report.reward_min,
                "max": report.reward_max,
            },
            "advantage_abs_mean": report.advantage_abs_mean,
            "protocol_error_rate": self._protocol_error_rate(groups),
            "terminated_by": self._terminated_by(groups),
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

        rows_summary = " ".join(f"{a}:{agent_stats[a]['n_rows']}" for a in AGENTS)
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
    def _check_degeneracy(self, report) -> None:
        cfg = self.config.magrpo
        if report.n_groups and report.degenerate_frac > cfg.max_degenerate_frac:
            self.degenerate_streak += 1
        else:
            self.degenerate_streak = 0
        if self.degenerate_streak >= cfg.degenerate_patience:
            raise RuntimeError(
                f"{self.degenerate_streak} consecutive iterations with "
                f">{cfg.max_degenerate_frac:.0%} degenerate groups: the reward is "
                "saturated or collapsed, so no gradient is being produced. "
                "Prescreen the pool or change the reward."
            )

    @staticmethod
    def _protocol_error_rate(groups) -> float:
        total = errors = 0
        for _, group in groups:
            for traj in group:
                total += len(traj.turns)
                errors += sum(1 for t in traj.turns if t.protocol_error)
        return errors / total if total else 0.0

    @staticmethod
    def _terminated_by(groups) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, group in groups:
            for traj in group:
                counts[traj.terminated_by] = counts.get(traj.terminated_by, 0) + 1
        return counts

    # -- resume -------------------------------------------------------------
    def save_state(self) -> None:
        self.logger.save_state(
            {
                "schema_version": 1,
                "iteration": self.iteration,
                "pool": self.pool.state(),
                "degenerate_streak": self.degenerate_streak,
                "elapsed_seconds": time.time() - self.started_at,
                "config_sha256": self.config.sha256(),
                "best": self.best,
                "adapters": {
                    a: str(self.adapters_dir / a / f"iter_{self.iteration - 1:04d}")
                    for a in AGENTS
                },
            }
        )

    def load_state(self) -> bool:
        state = self.logger.load_state()
        if not state:
            return False
        if state.get("config_sha256") != self.config.sha256():
            raise RuntimeError(
                "config changed since this run was created; resume would mix "
                "two different setups. Use a new run_id."
            )
        self.iteration = int(state["iteration"])
        self.pool.load_state(state["pool"])
        self.degenerate_streak = int(state.get("degenerate_streak", 0))
        self.best = state.get("best") or {}
        self.logger.print(f"resumed at iteration {self.iteration}")
        return True

    # -- driver -------------------------------------------------------------
    def run(
        self,
        *,
        max_iterations: int | None = None,
        ignore_wall_clock_limit: bool = False,
    ) -> None:
        budget = self.config.budget
        limit = max_iterations or budget.max_iterations
        deadline = None if ignore_wall_clock_limit else self.started_at + budget.max_wall_clock_hours * 3600
        while self.iteration < limit:
            if deadline is not None and time.time() > deadline:
                self.logger.print("wall-clock budget reached; stopping")
                break
            self.run_iteration()
            self._prune_adapters()

    def _prune_adapters(self) -> None:
        """Keep every save_every-th iteration plus the last few."""
        import shutil

        budget = self.config.budget
        for agent in AGENTS:
            directory = self.adapters_dir / agent
            if not directory.is_dir():
                continue
            checkpoints = sorted(
                (p for p in directory.iterdir() if p.is_dir() and p.name.startswith("iter_")),
                key=lambda p: p.name,
            )
            keep = set(checkpoints[-budget.keep_last :])
            for path in checkpoints:
                index = int(path.name.split("_")[1])
                if index % budget.save_every == 0:
                    keep.add(path)
            best_iter = (self.best or {}).get("iteration")
            if best_iter is not None:
                keep.add(directory / f"iter_{int(best_iter):04d}")
            for path in checkpoints:
                if path not in keep:
                    shutil.rmtree(path, ignore_errors=True)

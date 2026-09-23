"""MAPoRL's reward: ``score_rule`` + ``bonus_rule``.

Ported line-by-line from the official implementation. This module is the entire
contribution of the paper, so it is kept pure (plain floats, no torch) to make
every value hand-checkable in tests.

Provenance:
  score_rule   -- /tmp/maporl_ref/trl/trl/trainer/utils_multi_unified.py:1715
  bonus_rule   -- /tmp/maporl_ref/trl/trl/trainer/utils_multi_unified.py:1612
  composition  -- ppov2_trainer_multi_different_model.py:1607-1612

THREE PROPERTIES OF THE OFFICIAL CODE THAT SURPRISE PEOPLE. All three were
confirmed by executing the official functions, not by reading them:

1. ``bonus_rule`` is identically zero at turn 0. The ``if turn == 0: return``
   at :1628-1629 sits *before* both the backward block and the forward block
   (the latter at :1675). So the paper's "beta_0 = 2 improves Delta_0 by 17.2%"
   cannot come from a turn-0 bonus -- there is none. Turn-0 effects can only
   propagate through score_rule's discounted future term.

2. With three agents a 0.5 dead zone appears. "The others" are reduced to a
   *mean*, then compared with strict ``>`` / ``<`` against 0.5. Two agents give
   a mean in {0,1} so a branch always fires; three agents give {0, 0.5, 1}, and
   at exactly 0.5 neither branch fires and the whole backward bonus is dropped.
   This is our 2->3 agent change meeting a path the official code never ran.
   Hence ``others_tiebreak`` and the returned ``deadzone_*`` diagnostics.

3. The forward term's sign looks inverted relative to the paper. An agent that
   was correct while the others were wrong, and whose peers then became correct
   -- i.e. it successfully persuaded them -- is *penalised* ``alpha[3]``. Since
   alpha is validated non-negative upstream, no choice of alpha undoes it. We
   reproduce it as written (that is what "faithful" means) and expose
   ``forward_sign="corrected"`` for the alternative.
"""
from __future__ import annotations

from typing import Literal, Sequence

# utils_multi_unified.py:1618 -- "turn not approached" marker.
SENTINEL = -1.0
# :1645-1647 and siblings: nudge off the sentinel value after a bonus lands on it.
GUARD_EPS = 0.001

RuleHorizon = Literal["last", "discounted_sum", "current"]
RuleAgentShare = Literal["all", "individual"]
OthersTiebreak = Literal["dead", "majority"]
ForwardSign = Literal["official", "corrected"]

ScoreTable = dict[tuple[int, int], Sequence[float]]


def validate_alpha(alpha: Sequence[float]) -> list[float]:
    """Exactly four non-negative coefficients.

    The official entry script only checks non-empty and non-negative
    (``train_ppo_v2_multi_agent_multi_model.py:189-193``), so a shorter list
    reaches ``bonus_rule`` and IndexErrors on ``alpha[1]``. We reject early.
    """
    values = [float(a) for a in alpha]
    if len(values) != 4:
        raise ValueError(
            f"alpha must have exactly 4 components (got {len(values)}): "
            "[persuaded, self-generated, influence-same, influence-diff]"
        )
    if any(a < 0 for a in values):
        raise ValueError(f"alpha components must be non-negative: {values}")
    return values


def get_score_for_turn(
    table: ScoreTable,
    finished_question: Sequence[int],
    q: int,
    t: int,
    a: int,
) -> float:
    """Port of the nested helper at :1614-1624 / :1717-1725.

    Reproduces the official index remapping: rows exist only for questions still
    unfinished at turn ``t``, so ``q`` is looked up through the surviving
    indices. With no early stopping this is the identity, but it is kept so the
    sentinel path stays correct.
    """
    if finished_question[q] != -1 and t > finished_question[q]:
        return SENTINEL
    valid = [i for i, f in enumerate(finished_question) if f == -1 or f >= t]
    q_index = valid.index(q)
    return float(table[(t, a)][q_index])


def score_rule(
    *,
    rule_horizon: RuleHorizon,
    rule_agent_share: RuleAgentShare,
    discount_factor: float,
    scores_turn_agent: ScoreTable,
    total_rounds: int,
    total_agents: int,
    finished_question: Sequence[int],
    turn: int,
    agent: int,
) -> list[float]:
    """Influence-aware verification reward. Port of :1715-1798.

    The official GSM8k setting is ``("discounted_sum", "all")``:

        x   = total_rounds-1 if finished_question[q] == -1 else finished_question[q]
        D   = sum(d**(t'-turn) for t' in range(turn, x+1))
        fut = 0 if turn == total_rounds-1 else
              sum over t' in (turn, x] of d**(t'-turn) * mean_a s(q, t', a)
        out = (s(q, turn, agent) + fut) / D

    Note the inner mean spans ALL agents including ``agent`` itself (:1781-1787).
    """
    n_questions = len(finished_question)

    def s(q: int, t: int, a: int) -> float:
        return get_score_for_turn(scores_turn_agent, finished_question, q, t, a)

    if rule_horizon == "last" and rule_agent_share == "all":
        return [
            sum(s(q, total_rounds - 1, a) for a in range(total_agents)) / total_agents
            for q in range(n_questions)
        ]

    if rule_horizon == "last" and rule_agent_share == "individual":
        return [s(q, total_rounds - 1, agent) for q in range(n_questions)]

    if rule_horizon == "discounted_sum":
        out: list[float] = []
        for q in range(n_questions):
            x = total_rounds - 1 if finished_question[q] == -1 else finished_question[q]
            if turn > x:
                out.append(SENTINEL)
                continue
            discount_sum = sum(
                discount_factor ** (tp - turn) for tp in range(turn, x + 1)
            )
            if rule_agent_share == "all":
                current = s(q, turn, agent)
                if turn != total_rounds - 1:
                    future = sum(
                        sum(
                            discount_factor ** (tp - turn) * s(q, tp, a)
                            for a in range(total_agents)
                        )
                        / total_agents
                        for tp in range(turn + 1, x + 1)
                    )
                else:
                    future = 0.0
                out.append((current + future) / discount_sum)
            else:  # individual
                total = sum(
                    s(q, t, agent) * discount_factor ** (t - turn)
                    for t in range(turn, x + 1)
                )
                out.append(total / discount_sum)
        return out

    if rule_horizon == "current" and rule_agent_share == "individual":
        return [s(q, turn, agent) for q in range(n_questions)]

    if rule_horizon == "current" and rule_agent_share == "all":
        return [
            sum(s(q, turn, a) for a in range(total_agents)) / total_agents
            for q in range(n_questions)
        ]

    # Matches the official `else: raise ValueError("Rule not supported")` (:1796).
    raise ValueError(f"Rule not supported: {rule_horizon!r}/{rule_agent_share!r}")


def _above(value: float, threshold: float, tiebreak: OthersTiebreak) -> bool:
    if tiebreak == "majority":
        return value >= threshold
    return value > threshold


def _below(value: float, threshold: float, tiebreak: OthersTiebreak) -> bool:
    return value < threshold


def bonus_rule(
    *,
    correctnesses_turn_agent: ScoreTable,
    total_rounds: int,
    total_agents: int,
    finished_question: Sequence[int],
    turn: int,
    agent: int,
    alpha: Sequence[float],
    correct_threshold: float = 0.5,
    wrong_threshold: float = 0.5,
    others_tiebreak: OthersTiebreak = "dead",
    forward_sign: ForwardSign = "official",
) -> tuple[list[float], dict[str, int]]:
    """Alpha shaping on ground-truth correctness. Port of :1612-1717.

    Returns ``(bonus, diagnostics)``. The base value is 0.0 (or SENTINEL), NOT
    the correctness itself (:1626-1627).
    """
    a = validate_alpha(alpha)
    n_questions = len(finished_question)
    diag = {
        "backward_fired": 0,
        "forward_fired": 0,
        "deadzone_backward": 0,
        "deadzone_forward": 0,
        "guard_fired": 0,
    }

    def c(q: int, t: int, ag: int) -> float:
        return get_score_for_turn(correctnesses_turn_agent, finished_question, q, t, ag)

    # :1626-1627 -- base is zero, with the sentinel preserved where present.
    bonus = [
        SENTINEL if c(q, turn, agent) == SENTINEL else 0.0 for q in range(n_questions)
    ]

    # :1628-1629 -- the early return that makes turn 0 carry no bonus at all.
    if turn == 0:
        return bonus, diag

    def bump(q: int, delta: float) -> None:
        bonus[q] += delta
        # :1644-1646 etc. -- never let a bonus land exactly on the sentinel.
        if bonus[q] == SENTINEL:
            bonus[q] += GUARD_EPS
            diag["guard_fired"] += 1

    # --- backward: did this agent change, and what were the others doing? -----
    for q in range(n_questions):
        prev_self = c(q, turn - 1, agent)
        cur_self = c(q, turn, agent)
        others_prev = None
        if total_agents > 1:
            others_prev = sum(
                c(q, turn - 1, x) for x in range(total_agents) if x != agent
            ) / (total_agents - 1)

        if prev_self > correct_threshold and cur_self < wrong_threshold:
            if total_agents > 1:
                if _above(others_prev, correct_threshold, others_tiebreak):
                    bump(q, -a[1])  # generated a new wrong answer
                    diag["backward_fired"] += 1
                elif _below(others_prev, wrong_threshold, others_tiebreak):
                    bump(q, -a[0])  # got wrongly persuaded
                    diag["backward_fired"] += 1
                else:
                    diag["deadzone_backward"] += 1
            else:
                bump(q, -a[1])
                diag["backward_fired"] += 1

        if prev_self < wrong_threshold and cur_self > correct_threshold:
            if total_agents > 1:
                if _below(others_prev, wrong_threshold, others_tiebreak):
                    bump(q, a[1])  # self-corrected against the grain
                    diag["backward_fired"] += 1
                elif _above(others_prev, correct_threshold, others_tiebreak):
                    bump(q, a[0])  # correctly persuaded
                    diag["backward_fired"] += 1
                else:
                    diag["deadzone_backward"] += 1
            else:
                bump(q, a[1])
                diag["backward_fired"] += 1

    # --- forward: did this agent move the others? ----------------------------
    if turn < total_rounds - 1 and total_agents > 1:
        # sign = +1 reproduces the official code verbatim (:1692-1712); "corrected"
        # flips both branches so that persuading peers toward the truth is
        # rewarded rather than penalised. See module docstring item 3.
        sign = 1.0 if forward_sign == "official" else -1.0
        for q in range(n_questions):
            next_others = sum(
                c(q, turn + 1, x) for x in range(total_agents) if x != agent
            ) / (total_agents - 1)
            cur_self = c(q, turn, agent)
            cur_others = sum(
                c(q, turn, x) for x in range(total_agents) if x != agent
            ) / (total_agents - 1)

            # Peers went wrong -> correct. Official SUBTRACTS here.
            if _above(next_others, correct_threshold, others_tiebreak):
                if _below(cur_others, wrong_threshold, others_tiebreak):
                    if cur_self > correct_threshold:
                        bump(q, -sign * a[3])
                        diag["forward_fired"] += 1
                    elif cur_self < wrong_threshold:
                        bump(q, -sign * a[2])
                        diag["forward_fired"] += 1
                else:
                    diag["deadzone_forward"] += 1
            # Peers went correct -> wrong. Official ADDS here.
            elif _below(next_others, wrong_threshold, others_tiebreak):
                if _above(cur_others, correct_threshold, others_tiebreak):
                    if cur_self < wrong_threshold:
                        bump(q, sign * a[3])
                        diag["forward_fired"] += 1
                    elif cur_self > correct_threshold:
                        bump(q, sign * a[2])
                        diag["forward_fired"] += 1
                else:
                    diag["deadzone_forward"] += 1
            else:
                diag["deadzone_forward"] += 1

    return bonus, diag


def combine(score: Sequence[float], bonus: Sequence[float]) -> list[float | None]:
    """``score_rule + bonus_rule``, masked on score_rule's output.

    ``ppov2_trainer_multi_different_model.py:1607-1612``:
        scores_bonuses = score_rule(...) + bonus_rule(...)
        filtered       = scores_bonuses[scores != -1]

    The mask tests *score_rule's* output, not the sum. Masked cells are returned
    as ``None`` so callers drop them rather than train on a sentinel. Note the
    ``penalty_reward_value = -10`` is NOT caught by this mask -- it propagates
    into score_rule and then here.
    """
    if len(score) != len(bonus):
        raise ValueError(f"length mismatch: {len(score)} vs {len(bonus)}")
    return [
        None if s == SENTINEL else float(s) + float(b) for s, b in zip(score, bonus)
    ]


def shaped_rewards(
    *,
    scores_turn_agent: ScoreTable,
    correctnesses_turn_agent: ScoreTable,
    total_rounds: int,
    total_agents: int,
    finished_question: Sequence[int],
    alpha: Sequence[float],
    rule_horizon: RuleHorizon = "discounted_sum",
    rule_agent_share: RuleAgentShare = "all",
    discount_factor: float = 0.3,
    correct_threshold: float = 0.5,
    wrong_threshold: float = 0.5,
    others_tiebreak: OthersTiebreak = "dead",
    forward_sign: ForwardSign = "official",
) -> tuple[dict[tuple[int, int], list[float | None]], dict[str, int]]:
    """Every ``(turn, agent)`` cell's final reward, plus summed diagnostics.

    IMPORTANT: the official code replaces ``scores_turn_agent`` in place at
    :1613-1614, so every cell must be computed from the *original* table before
    anything is written back. This function reads only from its inputs and
    returns a fresh dict -- do not feed its output back in.
    """
    out: dict[tuple[int, int], list[float | None]] = {}
    totals = {
        "backward_fired": 0,
        "forward_fired": 0,
        "deadzone_backward": 0,
        "deadzone_forward": 0,
        "guard_fired": 0,
    }
    for turn in range(total_rounds):
        for agent in range(total_agents):
            score = score_rule(
                rule_horizon=rule_horizon,
                rule_agent_share=rule_agent_share,
                discount_factor=discount_factor,
                scores_turn_agent=scores_turn_agent,
                total_rounds=total_rounds,
                total_agents=total_agents,
                finished_question=finished_question,
                turn=turn,
                agent=agent,
            )
            bonus, diag = bonus_rule(
                correctnesses_turn_agent=correctnesses_turn_agent,
                total_rounds=total_rounds,
                total_agents=total_agents,
                finished_question=finished_question,
                turn=turn,
                agent=agent,
                alpha=alpha,
                correct_threshold=correct_threshold,
                wrong_threshold=wrong_threshold,
                others_tiebreak=others_tiebreak,
                forward_sign=forward_sign,
            )
            out[(turn, agent)] = combine(score, bonus)
            for key in totals:
                totals[key] += diag[key]
    return out, totals

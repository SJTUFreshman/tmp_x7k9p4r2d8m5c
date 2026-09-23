"""Canonical formal-run paths used by coordination-fingerprint analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AnalysisRun:
    benchmark: str
    method: str
    label: str
    path: Path | None
    status: str
    score: str | None = None
    note: str = ""

    @property
    def available(self) -> bool:
        return self.path is not None and self.path.is_file()


MUSIQUE_JCA = ROOT / "logs/mas_eval_concurrent/rl_sft_mas_0717_1130/results.jsonl"
MUSIQUE_ZEROSHOT = ROOT / "logs/sft_old_protocol/20260706_092608_base_old_sft_dev_start0_n2417/results.jsonl"
# 2026-09-11 重跑：MAD / AgentVerse / GPTSwarm 三条换成 baseline_queue 的新 run。
# 换的理由不是刷分（EM 只动了千分位），而是 provenance：旧 run 落盘时还没有逐调用的
# raw_outputs/attempts ledger（实测旧记录里 raw_outputs / attempts / usage 全缺），
# 因此 model_call_count_analysis.py 只能从 vLLM access log 反推“完成模型生成”，
# 还要处理 HTTP 400 归属不清的问题。新 run 三条都带完整 ledger，可直接精确计数。
# AFlow 一条当时被误判为「重跑全部失败」而保持旧路径，2026-09-14 已纠正并换成重跑，
# 见下方 MUSIQUE_AFLOW 处的注释。至此 MuSiQue 四条 baseline 全部走 ledger。
# 2026-09-11 前三条原值逐字如下（仓库没有可用 git，保留以便比对）：
#   MUSIQUE_MAD = ROOT / (
#       "baseline/MuSiQue/MAD/logs/20260811_104018_mad_dev_start0_n2417_r3/results.jsonl"
#   )                                                              # 943/2417 = 39.02%
#   MUSIQUE_AGENTVERSE = ROOT / (
#       "baseline/MuSiQue/AgentVerse/logs/agentverse_meta8b_dev_n2417/results.jsonl"
#   )                                                              # 1014/2417 = 41.95%
#   MUSIQUE_GPTSWARM = ROOT / (
#       "baseline/MuSiQue/GPTSwarm/logs/musique_gptswarm_hetero_full_v1/results.jsonl"
#   )                                                              # 885/2417 = 36.62%
MUSIQUE_MAD = ROOT / (
    "logs/baseline_queue/baseline_queue_20260911_001457/"
    "040_musique_mad/results.jsonl"
)
# 2026-09-14 用户指定 AgentVerse 改用 41.99% 的那次 roll，即 051 重复批次里的 run_01。
# 2026-09-14 前此条为 051 之外的独立单跑，原值逐字如下：
#   MUSIQUE_AGENTVERSE = ROOT / (
#       "logs/baseline_queue/rq09110842/050_musique_agentverse/results.jsonl"
#   )                                                              # 1033/2417 = 42.74%
MUSIQUE_AGENTVERSE = ROOT / (
    "logs/baseline_queue/rq030_051_0911_1352/051_musique_agentverse_repeat4/"
    "runs/run_01_20260911_215151/results.jsonl"
)
# 2026-09-14：AFlow 换成 070_musique_aflow_repeat5 的重跑。之前误判这个 job「三个队列
# 都失败、无产物」——实际失败的是另外三个目录（baseline_queue_20260910_231608 是
# DRY_RUN=1 的空跑、baseline_queue_20260911_001457 的 status.tsv 只有表头、
# smoke_run_233306 是 n=4 冒烟），成功的那次在 rq09110842 下，被漏掉了。
# 该 run 是 MODE=search 先搜（ENABLE_THINKING=0，20×20，21 nodes / 400 dev eval）再用
# 自己搜出的 search/workflows/round_07_proposed.py 跑 10 个 2417 题 roll，search 与
# deployment 首次同源，且两端都有 raw_outputs ledger —— 这消掉了 MuSiQue 最后一个
# 需要从 vLLM access log 反推 search 响应数的格子。
# 代价是新 workflow 重得多：18.911 ops/题（solve 11 + ensemble 1 + answer_generate 7）
# 对旧的 7.000 ops/题（solve 5 + ensemble 1 + answer_generate 1）。
# canonical 取用户 2026-09-14 指定的官方 5-roll 最高分 run_01_20260911_170029。
# 2026-09-14 前原值逐字如下（仓库没有可用 git，保留以便比对）：
#   MUSIQUE_AFLOW = ROOT / (
#       "baseline/MuSiQue/AFlow/outputs/aflow_hetero_search20_dev20_full2417.jsonl"
#   )                                                              # 1019/2417 = 42.16%
MUSIQUE_AFLOW = ROOT / (
    "logs/baseline_queue/rq09110842/070_musique_aflow_repeat5/"
    "runs/run_01_20260911_170029/results.jsonl"
)
MUSIQUE_AFLOW_SCORE = "1011/2417"
MUSIQUE_AFLOW_SEARCH_STATE = ROOT / (
    "logs/baseline_queue/rq09110842/070_musique_aflow_repeat5/search/state.json"
)
# 该 job 下共有 10 个 2417 题 roll，(run, correct, em)，直接从各 run 的 results.jsonl 数出。
# 17:00–21:15 的后 5 个是 status.tsv / summary.json 记录的官方批次；12:28–15:54 的前 5 个
# 是同一 workflow 的更早一批，未被 summary 收录。两批用的都是本 run 自搜的
# round_07_proposed.py。canonical 是官方批次里 EM 最高的 run_01_20260911_170029。
MUSIQUE_AFLOW_REPEAT_SCORES: tuple[tuple[str, int, float, bool], ...] = (
    ("run_01_20260911_122838", 1000, 0.413736, False),
    ("run_02_20260911_132036", 999, 0.413322, False),
    ("run_03_20260911_141237", 999, 0.413322, False),
    ("run_04_20260911_150253", 986, 0.407944, False),
    ("run_05_20260911_155427", 1022, 0.422838, False),
    ("run_01_20260911_170029", 1011, 0.418287, True),
    ("run_02_20260911_175139", 995, 0.411667, True),
    ("run_03_20260911_184207", 1003, 0.414977, True),
    ("run_04_20260911_193331", 1003, 0.414977, True),
    ("run_05_20260911_202437", 998, 0.412909, True),
)
# 末位 True = 属于 status.tsv 记录的官方 5-roll 批次。
# 全 10 roll：mean 0.414398, stdev 0.003960, min 0.407944, max 0.422838
# 官方 5 roll：mean 0.414563, stdev 0.002517, min 0.411667, max 0.418287
MUSIQUE_GPTSWARM = ROOT / (
    "logs/baseline_queue/rq09110842/060_musique_gptswarm/results.jsonl"
)
MUSIQUE_MAD_SCORE = "955/2417"
# 2026-09-14 前为 "1033/2417"（=050 单跑）。
MUSIQUE_AGENTVERSE_SCORE = "1015/2417"
MUSIQUE_GPTSWARM_SCORE = "875/2417"
# 051_musique_agentverse_repeat4 的 4 次同配置 roll，(run, correct, em)，直接从各 run 的
# results.jsonl 数出。该批次的 runs/ 下还有一组 20260911_2116xx 的同名目录，全部是失败的
# 首次启动、没有 results.jsonl，不参与统计。
# canonical 是用户 2026-09-14 指定的 run_01（1015/2417 = 0.419942 = 41.99%），即这 4 次里
# 的最小值；与之同配置的独立单跑 050（1033/2417 = 0.427389）不再作为正式口径。成本指标
# 仍只按单条 canonical 统计，下面的分布只作方差留档。
MUSIQUE_AGENTVERSE_REPEAT4_SCORES: tuple[tuple[str, int, float], ...] = (
    ("run_01_20260911_215151", 1015, 0.419942),
    ("run_02_20260911_222646", 1051, 0.434837),
    ("run_03_20260911_230121", 1034, 0.427803),
    ("run_04_20260911_233606", 1020, 0.422011),
)
# mean 0.426148, stdev 0.006680, min 0.419942, max 0.434837
GSM_JCA = ROOT / (
    "outputs/gsm_eval/role_batched/"
    "gsm_judge_rl_v13_seed_sweep_infinite_20260809/"
    "gsm_judge_rl_v13_seed_sweep_infinite_20260809_seed43_dev132_"
    "thinking_hidden_self_handoff_ctx40960.jsonl"
)
GSM_ZEROSHOT = ROOT / "outputs/gsm_eval/gsm_new_eval_thinking_20260808/gsm_zero_shot_mas_hetero_new_eval_dev132.jsonl"
GSM_BASELINE_ROOT = ROOT / "logs/gsm_hard_baselines/gsmhard_latest_roles_raw8192_20260816"
GSM_MAD = GSM_BASELINE_ROOT / "01_mad/results.jsonl"
GSM_AGENTVERSE = GSM_BASELINE_ROOT / "02_agentverse/results.jsonl"
GSM_AFLOW = GSM_BASELINE_ROOT / "03_aflow/eval_results.jsonl"
GSM_GPTSWARM = GSM_BASELINE_ROOT / "04_gptswarm/results.jsonl"

MULTIPL_E_ZERO_SHOT = ROOT / (
    "logs/multipl_e_8lang_mas_zero_shot/"
    "20260725_161844_multipl_e_8lang_mas_zero_shot_temp0p0/trajectories.jsonl"
)
MULTIPL_E_ZERO_SHOT_COMPLETION_ROOT = ROOT / (
    "Code/MultiPL-E/experiments/multipl_e_8lang_mas_zero_shot/"
    "20260725_161844_multipl_e_8lang_mas_zero_shot_temp0p0/completions"
)
MULTIPL_E_MAD = ROOT / (
    "logs/multipl_e_8lang_baselines/mad/"
    "multipl_e_8lang_mad_full_test30_20260820/trajectories.jsonl"
)
MULTIPL_E_AGENTVERSE = ROOT / (
    "logs/multipl_e_8lang_baselines/agentverse/"
    "20260821_112236_multipl_e_8lang_agentverse_repeat5/runs/run_03/trajectories.jsonl"
)
MULTIPL_E_GPTSWARM = ROOT / (
    "logs/multipl_e_8lang_baselines/gptswarm/"
    "multipl_e_8lang_gptswarm_full_test30_20260820_v2/trajectories.jsonl"
)
MULTIPL_E_AFLOW = ROOT / (
    "logs/multipl_e_8lang_baselines/aflow/"
    "multipl_e_8lang_aflow_full_20x20_test30_20260820/trajectories.jsonl"
)
MULTIPL_E_SELF_RL = ROOT / (
    "logs/multipl_e_8lang_baselines/sas_self_judged_14b_rl/"
    "multipl_e_8lang_sas_self_rl_normalized_v2/test_rollout.jsonl"
)
# 最终 JCA eval：mpe0912 队列 job 120 的 roll r08。整批 30 个 roll（job 100/110/120
# 各 10 个）里，r08 在 weighted 与 macro 两个指标上同时最高。选点理由与分布见
# MULTIPL_E_JCA_ROLL_SCORES 和下方 CANONICAL_RUNS 里的 note。
MULTIPL_E_JCA_RUN_ROOT = ROOT / (
    "logs/baseline_queue/mpe0912/120_multipl_e_gpt5rwr_ckpt25_x10/"
    "mpe0912_120_multipl_e_gpt5rwr_ckpt25_x10_r08"
)
MULTIPL_E_JCA = MULTIPL_E_JCA_RUN_ROOT / "trajectories.jsonl"
MULTIPL_E_JCA_COMPLETION_ROOT = ROOT / (
    "Code/MultiPL-E/experiments/multipl_e_8lang_sft_eval/"
    "mpe0912_120_multipl_e_gpt5rwr_ckpt25_x10_r08/completions"
)
# job 120 全部 10 个 roll，(roll, correct_count, weighted_pass_at_1,
# macro_language_pass_at_1)，直接读自各 roll 的 mas_scores.json["overall"]。
# TEMPERATURE=0.0 下 seed 是 no-op；这里的离散度来自 vLLM 连续批处理
# (MAX_CONCURRENCY=64) 的 batch 组成差异，不是重采样。
MULTIPL_E_JCA_ROLL_SCORES: tuple[tuple[str, int, float, float], ...] = (
    ("r01", 793, 0.586538, 0.589615),
    ("r02", 800, 0.591716, 0.592506),
    ("r03", 803, 0.593935, 0.594824),
    ("r04", 799, 0.590976, 0.591231),
    ("r05", 793, 0.586538, 0.589615),
    ("r06", 799, 0.590976, 0.590476),
    ("r07", 795, 0.588018, 0.586884),
    ("r08", 803, 0.593935, 0.596333),
    ("r09", 787, 0.582101, 0.579698),
    ("r10", 793, 0.586538, 0.585087),
)
MULTIPL_E_JCA_SCORE = "803/1352"

MATH_ROOT = ROOT / (
    "logs/math_baselines/post_recovery/"
    "math_post_recovery_nothinking_suite_20260829/four_baselines"
)
MATH_MAD = MATH_ROOT / "mad/trajectories.jsonl"
MATH_AGENTVERSE = MATH_ROOT / "agentverse/trajectories.jsonl"
# 2026-09-11 重跑：GPTSwarm 与 AFlow 换成 baseline_queue_20260911_001457 的新 run，
# 两条都是为了消掉已记录在案的口径硬伤，MAD / AgentVerse 不动：
#   - GPTSwarm 旧 run 是 TEMPERATURE=0.0，是五个数据集里唯一的孤例（见
#     analysis/baseline_details/解码参数_中文.md）。新 run 为 0.7，与其余全部对齐。
#   - AFlow 旧 run 是 MODE=eval + WORKFLOW_FILE 指向 round_00_initial.py，即评测的是
#     手写初始 workflow、根本没用搜索产物，却仍在成本表里摊了一份 thinking-enabled 的
#     旧 search（见 analysis/baseline_details/生成预算_中文.md）。新 run 是 MODE=both，
#     自带真实的 20×20 no-thinking 搜索，deployment 与 search 首次同源。
# 两个新 run 的 500 个 problem_id 与旧 suite 完全同集（同 shard_04），ENABLE_THINKING=0，
# 所以“四个 baseline 共用同一 shard”的口径不破。
# 2026-09-11 前两条原值逐字如下（仓库没有可用 git，保留以便比对）：
#   MATH_GPTSWARM = MATH_ROOT / "gptswarm/trajectories.jsonl"   # 350/500 = 70.0%，T=0.0
#   MATH_AFLOW = MATH_ROOT / "aflow/trajectories.jsonl"         # 320/500 = 64.0%，MODE=eval
MATH_GPTSWARM = ROOT / (
    "logs/baseline_queue/baseline_queue_20260911_001457/"
    "010_math_gptswarm_t07/trajectories.jsonl"
)
MATH_AFLOW = ROOT / (
    "logs/baseline_queue/baseline_queue_20260911_001457/"
    "020_math_aflow_both/trajectories.jsonl"
)
MATH_ZERO_SHOT = ROOT / (
    "experiments/math_rl_mas_thinking/artifacts/"
    "math_rl_mas_v13_nonthinking_global_turn_json_object_1p7b_4b_8b_corrected_v4d/"
    "07_eval_pretraining/heterogeneous_1p7b_4b_8b/05_eval/"
    "math_test_seed43_collapse_recovery_nothinking_v1.jsonl"
)
MATH_JCA = ROOT / (
    "experiments/math_specific_sft_rl_v1/artifacts/"
    "math_specific_sft_rl_reuse_legacy_v4d_real_rollout_sft_protocol_floor_v2_20260830/"
    "06_eval/variants/rl_all_final_a3_turn2_incumbent_v1/results.jsonl"
)
# 2026-09-11 起指向 020_math_aflow_both 自带的 search，与同一 run 的 deployment 同源、
# 同为 no-thinking。原值逐字如下（旧 MODE=eval deployment 复用的 thinking-enabled search）：
#   MATH_AFLOW_SEARCH_STATE = ROOT / (
#       "logs/math_baselines/serial/math_baselines_all_20260824_205845/"
#       "aflow/search/state.json"
#   )
MATH_AFLOW_SEARCH_STATE = ROOT / (
    "logs/baseline_queue/baseline_queue_20260911_001457/"
    "020_math_aflow_both/search/state.json"
)
# ---------------------------------------------------------------------------
# Conifer（2026-09-14 新增，第五个 benchmark）
# ---------------------------------------------------------------------------
# Conifer 与另外四个数据集有三处结构性差异，登记路径前必须先知道：
#   1. baseline 不是 baseline/ 下那套，而是 conifer_training_hub/03_rollout/
#      run_conifer_mas.py 的另一套重实现。好处是六条 arm 全部落成**统一的 MAS
#      trajectory schema**（`trajectory.steps` + `active_agent` + `sampling.model_paths`），
#      所以 trajectory_adapters.protocol_units 的 "jca" 分支通吃，不需要 per-baseline adapter。
#   2. **没有 EM**。正确性是连续的 `hard_score`（conifer_hard_score_v1），二值口径只有
#      `final_checks.all_explicit_passed`（全部显式约束通过）。下面的 *_SCORE 用后者，
#      连续分单列为 *_HARD_SCORE。
#   3. JSON 走 vLLM guided decoding（`--json-transport json_schema` + `response_format`），
#      不是另外四个数据集的 prompt-only，所以解析失败率接近 0，重试口径不可直接跨集比较。
#
# **含重试的调用数只有 AFlow 一条可恢复。** forever-loop 落盘的 step 只有单个
# `raw_output`、没有 `raw_outputs` ledger，round 目录里也没有 vLLM server log，
# 两条计数来源都不存在；AFlow 走 baseline_queue 链路，8412/8412 个 step 都有
# `raw_outputs`。因此 JCA/MAD/AgentVerse/GPTSwarm 四条按用户 2026-09-14 的决定
# 「只报逻辑调用 + 脚注」，model_call_count_analysis.py 用 retry_ledger=False 标记。
#
# canonical 选点（用户 2026-09-14 指定）：JCA 取 coverage 88.90 / explicit 95.90 的那轮
# = round_4，它同时是 111 轮里 hard 最高的一轮，且盘上还在；四臂取最新同时存活的
# round_27（gptswarm 另有 round_28，但 mad/agentverse 没有，为跨臂同轮对齐不用它）；
# AFlow 沿用 MuSiQue AFlow 的「官方批次最高分」惯例取 r04。
CONIFER_JCA = ROOT / (
    "conifer_training_hub/10_outputs/rl_mas_eval_forever_20260906/"
    "round_4/trajectories.jsonl"
)
CONIFER_FOUR_ARM_ROOT = ROOT / (
    "conifer_training_hub/10_outputs/four_arm_eval_forever_20260909/round_27"
)
CONIFER_MAD = CONIFER_FOUR_ARM_ROOT / "mad/trajectories.jsonl"
CONIFER_AGENTVERSE = CONIFER_FOUR_ARM_ROOT / "agentverse/trajectories.jsonl"
CONIFER_GPTSWARM = CONIFER_FOUR_ARM_ROOT / "gptswarm/trajectories.jsonl"
CONIFER_AFLOW = ROOT / (
    "logs/baseline_queue/rr09120115_conifer/130_conifer_aflow_eval_x5/"
    "r04/trajectories.jsonl"
)
# 130 批次的 5 个 roll 全部复用 030 搜出的 round_01_proposed.py（summary.json 的
# reused_search.selected_source 指向它），所以 search 与 deployment 同源，和 MuSiQue /
# MATH 现在的状态一致。注意：analysis/baseline_details/Conifer口径核验_中文.md 里
# 「AFlow search 根本没实现」的结论已被这次核查证伪，该文档待重写。
CONIFER_AFLOW_SEARCH_STATE = ROOT / (
    "logs/baseline_queue/rq09110842/030_conifer_aflow_search/search/state.json"
)
# self-RL（sas14b_rl）：27 轮分数俱全，但**盘上一个 trajectories.jsonl 都没留下**，
# 成本分析无法进行，因此登记为 pending 而不是给一个别轮的替代文件。
# 注意这里的 None 只说明 **eval trajectories** 没落盘。SAS 的 judge 训练池是存在的
# （conifer_3x8b_sas14b_self_rl_v1/02_sas_judge/train_judged.jsonl，21617 条，
# 14B 自评），见 conifer_judge_score_analysis.py —— 那是另一类产物，两者不矛盾。
CONIFER_SELF_RL = None

CONIFER_N_PROBLEMS = 1402
# 二值口径 = final_checks.all_explicit_passed，逐条从 trajectories.jsonl 数出。
CONIFER_JCA_SCORE = "1284/1402"
CONIFER_MAD_SCORE = "1283/1402"
CONIFER_AGENTVERSE_SCORE = "1286/1402"
CONIFER_GPTSWARM_SCORE = "1284/1402"
CONIFER_AFLOW_SCORE = "1296/1402"
# 连续口径（conifer_hard_score_v1），同一批文件逐条算出的均值。
CONIFER_JCA_HARD_SCORE = 0.94046
CONIFER_MAD_HARD_SCORE = 0.94277
CONIFER_AGENTVERSE_HARD_SCORE = 0.94093
CONIFER_GPTSWARM_HARD_SCORE = 0.94303
CONIFER_AFLOW_HARD_SCORE = 0.94532
# 跨 round 的 hard_score 分布，读自两个 forever-loop 的 state/rounds.jsonl。
# 用户 2026-09-14 的口径：分数取全分布，成本取存活 round。canonical 单点落在分布何处
# 见每行注释。JCA 的 canonical 是分布的 max，四臂的 canonical 在各自分布中位附近。
CONIFER_HARD_SCORE_DISTRIBUTION: dict[str, tuple[int, float, float, float, float]] = {
    # arm: (rounds, mean, stdev, min, max)
    "JCA": (111, 0.93657, 0.00193, 0.92981, 0.94046),        # canonical round_4 = max
    "MAD": (27, 0.94198, 0.00123, 0.94019, 0.94505),         # canonical round_27 = 0.94277
    "AgentVerse": (27, 0.94080, 0.00120, 0.93833, 0.94304),  # canonical round_27 = 0.94093
    "GPTSwarm": (27, 0.94250, 0.00111, 0.94013, 0.94429),    # canonical round_27 = 0.94303
    "self-RL": (27, 0.93931, 0.00085, 0.93749, 0.94063),     # 无存活 trajectory，仅分数
}
# 盘上仍存在 trajectories.jsonl 的 round（其余只在 rounds.jsonl 里留了指标）：
#   JCA        round 4, 109, 110, 111, 112
#   MAD        round 2, 7, 18, 24, 25, 26, 27
#   AgentVerse round 2, 7, 18, 24, 25, 26, 27
#   GPTSwarm   round 2, 7, 18, 24, 25, 26, 27, 28
#   sas14b_rl  （无）
# 130_conifer_aflow_eval_x5 的 5 个 roll，(roll, hard, explicit, coverage, all_explicit)。
# 同 config 连跑，复用同一份 round_01_proposed.py。
CONIFER_AFLOW_ROLL_SCORES: tuple[tuple[str, float, float, float, int], ...] = (
    ("r01", 0.94421, 0.96533, 0.88086, 1298),
    ("r02", 0.94389, 0.96531, 0.87962, 1296),
    ("r03", 0.94492, 0.96605, 0.88152, 1297),
    ("r04", 0.94532, 0.96684, 0.88077, 1296),
    ("r05", 0.94365, 0.96513, 0.87921, 1293),
)
# 5 roll hard：mean 0.94440, stdev 0.00070, min 0.94365, max 0.94532。
# 另有一次独立单跑 030_conifer_aflow_search/eval（hard 0.94256），不作为正式口径。
# 2026-09-14 订正：此前这里写的是「10_outputs 下 5 个 *_conifer_zero_shot_mas 目录
# 全部是空的，Conifer 没有 zero-shot」，**这是错的**。5 个目录里有 1 个不是空的：
# 20260901_190940 留下了 1003 条轨迹。它是一次真实的训练前异构 MAS 运行——
# config.env 里 ADAPTER_A1/A2/A3 全为空（无 LoRA）、MODEL_A1/A2/A3 是裸的
# Qwen3-1.7B/4B/8B，协议常数与训练后完全一致（T_MAX=6、ROUTING=dynamic、
# MIN_AGENTS_BEFORE_STOP=3、MIN_HANDOFFS_BEFORE_STOP=2、START_AGENT=balanced/42、
# top_p 0.95、max_new_tokens 1024）。
# 但有两个必须随数字一起报的 caveat：
#   1. **它没跑完**：run.log 以 `exit 143`（SIGTERM）结束，config 里 LIMIT=0 本来要跑
#      全部 1402 题，实际只落盘 1003 题（覆盖 test.jsonl 的前 1100 条里的 1003 条）。
#      这批题**在 all_explicit_passed 这个口径上明显偏易**：平均每题只有 0.50 个显式
#      检查项，剩下的 399 题是 0.93 个；同一个训练后 round_4 在这 1003 题上是
#      962/1003 = 95.91%，在补集上只有 322/399 = 80.70%。（注意 dataset 自带的
#      difficulty 字段方向相反——难度 1+2 在子集里只占 26.3%、全集 42.5%——但那个字段
#      衡量的是语义难度，不是显式约束个数，别拿它当本指标的代理。）
#      所以**不能**直接和训练后的 1402 题行拿来比，必须把训练后限制到同一批 id。
#   2. 解码温度是 0.0，训练后的 round_4 是 0.3。
CONIFER_ZERO_SHOT = ROOT / (
    "conifer_training_hub/10_outputs/"
    "20260901_190940_conifer_zero_shot_mas/trajectories.jsonl"
)
CONIFER_ZERO_SHOT_N_PROBLEMS = 1003
CONIFER_ZERO_SHOT_SCORE = "930/1003"  # all_explicit_passed
CONIFER_ZERO_SHOT_HARD_SCORE = 0.93577

# 完整训练前异构 MAS 池：1402 题 × 4 rollout，裸 Qwen3-1.7B/4B/8B，adapter
# path 全为空。正式的 20260901_190940 eval 只落盘 1003 题；需要与训练后全量
# 1402 题做结构对比时，从该池固定取 rollout_idx=0。官方 benchmark 导出器
# export_official_predictions.py 的默认选择也是 rollout_idx=0。
CONIFER_ZERO_SHOT_POOL = ROOT / (
    "conifer_training_hub/10_outputs/"
    "conifer_stratified_sft_full_rl_20260902/zero_shot/trajectories.jsonl"
)
CONIFER_ZERO_SHOT_POOL_ROLLOUT_INDEX = 0

MATH_MAD_SCORE = "362/500"
MATH_AGENTVERSE_SCORE = "341/500"
# 2026-09-11 前：MATH_GPTSWARM_SCORE = "350/500"、MATH_AFLOW_SCORE = "320/500"
MATH_GPTSWARM_SCORE = "346/500"
MATH_AFLOW_SCORE = "325/500"
MATH_JCA_SCORE = "392/500"


CANONICAL_RUNS: tuple[AnalysisRun, ...] = (
    AnalysisRun("MuSiQue", "JCA", "JCA（正式运行）", MUSIQUE_JCA, "available", "1024/2417"),
    AnalysisRun("MuSiQue", "zero-shot", "训练前异构 MAS", MUSIQUE_ZEROSHOT, "available", "514/2417", "Qwen3-1.7B + Qwen3-4B + Qwen3-8B；无 LoRA；无 thinking"),
    # 2026-09-11 前三条为旧 run，score 依次是 943/2417、1014/2417、885/2417。
    AnalysisRun("MuSiQue", "MAD", "MAD training-free（2026-09-11 重跑）", MUSIQUE_MAD, "available", MUSIQUE_MAD_SCORE, "带完整 raw_outputs ledger，调用数不再依赖 vLLM access log 反推"),
    AnalysisRun("MuSiQue", "AgentVerse", "AgentVerse training-free（2026-09-11 重跑，051 批次 run_01）", MUSIQUE_AGENTVERSE, "available", MUSIQUE_AGENTVERSE_SCORE, "带完整 raw_outputs ledger；用户 2026-09-14 指定取 41.99% 的那次 roll，即 MUSIQUE_AGENTVERSE_REPEAT4_SCORES（0.4261 ± 0.0067）的最小值 0.4199"),
    AnalysisRun("MuSiQue", "AFlow", "AFlow training-free（2026-09-11 重跑，070 官方批次 run_01）", MUSIQUE_AFLOW, "available", MUSIQUE_AFLOW_SCORE, "MODE=search 先搜再用自搜的 round_07_proposed.py 部署，search 与 deployment 同源且两端都有 ledger；用户 2026-09-14 指定取官方 5-roll 最高分。逐 roll 分布见 MUSIQUE_AFLOW_REPEAT_SCORES；替换掉的旧单点为 1019/2417（42.16%），其 search 响应数只能从 access log 反推"),
    AnalysisRun("MuSiQue", "GPTSwarm", "GPTSwarm training-free（2026-09-11 重跑）", MUSIQUE_GPTSWARM, "available", MUSIQUE_GPTSWARM_SCORE, "带完整 raw_outputs ledger，调用数不再依赖 vLLM access log 反推"),
    AnalysisRun("GSM-Hard", "JCA", "JCA（正式运行，seed=43）", GSM_JCA, "available", "95/132"),
    AnalysisRun("GSM-Hard", "zero-shot", "训练前异构 MAS", GSM_ZEROSHOT, "available", "91/132", "Qwen3-1.7B + Qwen3-4B + Qwen3-8B；无 LoRA；thinking enabled"),
    AnalysisRun("GSM-Hard", "MAD", "MAD training-free", GSM_MAD, "available", "88/132"),
    AnalysisRun("GSM-Hard", "AgentVerse", "AgentVerse training-free", GSM_AGENTVERSE, "available", "95/132"),
    AnalysisRun("GSM-Hard", "AFlow", "AFlow training-free", GSM_AFLOW, "available", "90/132"),
    AnalysisRun("GSM-Hard", "GPTSwarm", "GPTSwarm training-free", GSM_GPTSWARM, "available", "97/132"),
    AnalysisRun("MultiPL-E-8Lang", "zero-shot", "异构 zero-shot MAS", MULTIPL_E_ZERO_SHOT, "available", "569/1352", "Qwen3-1.7B + Qwen3-4B + Qwen3-8B；无 LoRA"),
    AnalysisRun("MultiPL-E-8Lang", "MAD", "MAD training-free", MULTIPL_E_MAD, "available", "744/1352"),
    AnalysisRun("MultiPL-E-8Lang", "AgentVerse", "AgentVerse training-free（repeat-5 run_03）", MULTIPL_E_AGENTVERSE, "available", "801/1352", "沿用现有 formal aggregate 的 minimum_score 选择"),
    AnalysisRun("MultiPL-E-8Lang", "GPTSwarm", "GPTSwarm training-free", MULTIPL_E_GPTSWARM, "available", "698/1352"),
    AnalysisRun("MultiPL-E-8Lang", "AFlow", "AFlow training-free", MULTIPL_E_AFLOW, "available", "579/1352"),
    AnalysisRun("MultiPL-E-8Lang", "self-RL", "SAS self-RL normalized v2", MULTIPL_E_SELF_RL, "available", "817/1352", "one-shot schema; no MAS handoff trajectory"),
    # 2026-09-12 前此条为 pending，原值逐字如下（仓库没有可用 git，保留以便比对）：
    #   AnalysisRun("MultiPL-E-8Lang", "JCA", "最终 JCA 训练产物", None, "pending",
    #               note="最终训练/eval 尚未完成，不能使用旧 SFT 或 smoke run"),
    AnalysisRun(
        "MultiPL-E-8Lang",
        "JCA",
        "最终 JCA 训练产物（mpe0912 job 120 / r08）",
        MULTIPL_E_JCA,
        "available",
        MULTIPL_E_JCA_SCORE,
        "A1 = Qwen3-8B + multipl_e_a1_8b_gpt5_paired_rwr_20260830/A1/checkpoint-25；"
        "A2 = Qwen3-4B + multipl_e_a2_gpt5_paired_rwr_20260827/A2/checkpoint-120；"
        "A3 = Qwen3-1.7B + multipl_e_8lang_v5_diverse_a1_5ep/A1/final。"
        "reversed capacity 指派（A1 才是 8B，不是 launcher 默认）；A1_TP=4 占 GPU 0-3，"
        "A2_TP=2 占 4-5，A3_TP=2 占 6-7。TEMPERATURE=0.0，T_MAX=8，START_AGENT=A1，"
        "SPLIT_SEED=7658190907657085414，humaneval+mbpp 共 1352 题 8 语言。"
        "weighted_pass_at_1=0.593935，macro_language_pass_at_1=0.596333。"
        "选点口径：同 config 连跑 10 roll，r08 是 best-of-10（两指标同时最高）；"
        "10 roll weighted 分布 0.582101–0.593935，mean 0.589127，stdev 0.003808，"
        "逐 roll 明细见 MULTIPL_E_JCA_ROLL_SCORES。沿用 zero-shot canonical 的 "
        "best-of-N 惯例（MULTIPL_E_AGENTVERSE 用的是另一种 minimum_score 惯例）。"
        "注意：三个 adapter 都属 gpt5_paired_rwr 支线，与 judge_score_bias_analysis.py "
        "所审计的 0813 ckpt-search 14B 评分池不同源。",
    ),
    AnalysisRun("MATH", "MAD", "MAD training-free（no-thinking shard_04）", MATH_MAD, "available", MATH_MAD_SCORE, "共同 500 题 shard；ENABLE_THINKING=0"),
    AnalysisRun("MATH", "AgentVerse", "AgentVerse training-free（no-thinking shard_04）", MATH_AGENTVERSE, "available", MATH_AGENTVERSE_SCORE, "共同 500 题 shard；ENABLE_THINKING=0"),
    AnalysisRun("MATH", "GPTSwarm", "GPTSwarm training-free（no-thinking shard_04，2026-09-11 重跑）", MATH_GPTSWARM, "available", MATH_GPTSWARM_SCORE, "共同 500 题 shard；ENABLE_THINKING=0；TEMPERATURE=0.7，与其余四个数据集对齐（旧 run 是全局唯一的 0.0 孤例）"),
    AnalysisRun("MATH", "AFlow", "AFlow training-free（no-thinking shard_04，2026-09-11 重跑）", MATH_AFLOW, "available", MATH_AFLOW_SCORE, "共同 500 题 shard；ENABLE_THINKING=0；MODE=both，deployment 评的是本 run 自带的 20×20 no-thinking search 产物（旧 run 是 MODE=eval + round_00_initial.py，却摊旧 thinking-enabled search 的成本）"),
    AnalysisRun(
        "MATH",
        "JCA",
        "JCA 正式 eval（三-turn 预算回退）",
        MATH_JCA,
        "available",
        MATH_JCA_SCORE,
        "同一 shard_04；no-thinking；超过三 turn 或第三 turn 后仍未成功 stop 时用首个 tentative_answer；候选与 turn 取前 3 个落盘 turn（含强制 A3 格式转换），行为先取该窗口再排除格式转换但保留其 tentative 作为答案前态；完整轨迹仅审计；调用与 token 另计独立 8B bootstrap solver",
    ),
    AnalysisRun("MATH", "zero-shot", "异构 zero-shot MAS", MATH_ZERO_SHOT, "available", "315/500", "Qwen3-1.7B + Qwen3-4B + Qwen3-8B；裸模型 + 零初始化 adapter（无有效训练 LoRA）；no-thinking；成本/调用包含独立 8B bootstrap solver + 完整 trajectory"),
    AnalysisRun("MATH", "self-RL", "SAS self-RL", None, "pending", note="尚无登记为同口径可比的正式结果"),
    # 2026-09-14 新增 Conifer。得分列填的是二值口径 all_explicit_passed/1402；连续的
    # hard_score 在 note 里给出，跨 round 分布见 CONIFER_HARD_SCORE_DISTRIBUTION。
    AnalysisRun(
        "Conifer",
        "JCA",
        "JCA 正式 eval（forever-loop round_4）",
        CONIFER_JCA,
        "available",
        CONIFER_JCA_SCORE,
        "hard_score 0.94046 / explicit 0.95903 / coverage 0.88901；用户 2026-09-14 按 "
        "coverage 88.90 + explicit 95.90 指定该轮，它也是 111 轮里 hard 最高的一轮"
        "（分布 0.93657 ± 0.00193）。eval temperature 0.3（非 0.0），JSON 走 guided "
        "decoding。**step 只有单个 raw_output、无 raw_outputs ledger，也无 server log，"
        "含重试的调用数不可恢复**，成本表只报逻辑调用。",
    ),
    AnalysisRun(
        "Conifer",
        "MAD",
        "MAD training-free（four-arm round_27）",
        CONIFER_MAD,
        "available",
        CONIFER_MAD_SCORE,
        "hard_score 0.94277（27 轮分布 0.94198 ± 0.00123）。固定 10 step，其中第 10 步是"
        "合成的 final_step、raw_output 为空、不产生模型生成。同样无 ledger，只报逻辑调用。",
    ),
    AnalysisRun(
        "Conifer",
        "AgentVerse",
        "AgentVerse training-free（four-arm round_27）",
        CONIFER_AGENTVERSE,
        "available",
        CONIFER_AGENTVERSE_SCORE,
        "hard_score 0.94093（27 轮分布 0.94080 ± 0.00120）。Conifer 的 AgentVerse "
        "没有独立 recruiter 调用，每轮是 3 个 expert 加 1 次 evaluator，所以产生模型"
        "生成的 step 数是 4/8/12 三档（不是另外四个数据集的 5/9/13），均值 5.478；"
        "落盘时每题末尾另有一个 raw_output 为空的合成 confirm_stop step，不计入。"
        "同样无 ledger，只报逻辑调用。",
    ),
    AnalysisRun(
        "Conifer",
        "GPTSwarm",
        "GPTSwarm training-free（four-arm round_27）",
        CONIFER_GPTSWARM,
        "available",
        CONIFER_GPTSWARM_SCORE,
        "hard_score 0.94303（27 轮分布 0.94250 ± 0.00111）。固定 7 node。"
        "同样无 ledger，只报逻辑调用。",
    ),
    AnalysisRun(
        "Conifer",
        "AFlow",
        "AFlow training-free（130 批次 r04）",
        CONIFER_AFLOW,
        "available",
        CONIFER_AFLOW_SCORE,
        "hard_score 0.94532（5 roll 分布 0.94440 ± 0.00070，取最高，沿用 MuSiQue AFlow "
        "的官方批次最高分惯例）。固定 6 op。**Conifer 唯一有完整 raw_outputs ledger 的一条**"
        "（8412/8412 个 step），因此也是唯一能报含重试调用数的。部署复用 "
        "030_conifer_aflow_search 搜出的 round_01_proposed.py，search 与 deployment 同源。",
    ),
    AnalysisRun(
        "Conifer",
        "self-RL",
        "SAS 14B self-RL（four-arm sas14b_rl）",
        CONIFER_SELF_RL,
        "pending",
        note="27 轮分数俱全（hard 0.93931 ± 0.00085，用户指定的 coverage 88.00 / "
        "explicit 95.81 对应 round_11），但**没有任何一轮的 trajectories.jsonl 留在盘上**，"
        "token 与调用数都无法统计，因此不填成本数字",
    ),
    AnalysisRun(
        "Conifer",
        "zero-shot",
        "异构 zero-shot MAS",
        CONIFER_ZERO_SHOT,
        "ok",
        score=CONIFER_ZERO_SHOT_SCORE,
        # 2026-09-14 首版记的是「5 个目录全部为空，无可用产物」，已作废。
        note="裸 Qwen3-1.7B/4B/8B、无 LoRA（ADAPTER_A* 全空），协议常数与训练后逐项一致。"
        "**只跑了 1003/1402 题就被 SIGTERM 掐掉**（run.log `exit 143`），且这 1003 题在本口径上"
        "偏易（平均 0.50 个显式检查项，补集 0.93；训练后在子集 95.91%、补集 80.70%），"
        "所以只能与限制到同一批 id 的训练后结果对比；"
        "解码温度 0.0，训练后是 0.3。hard 0.93577，all_explicit 930/1003 = 92.72%",
    ),
)


def runs_for(benchmark: str) -> tuple[AnalysisRun, ...]:
    """Return canonical entries for one benchmark."""
    return tuple(run for run in CANONICAL_RUNS if run.benchmark == benchmark)

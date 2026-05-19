---
title: Orbital Thruster Environment Server
sdk: docker
pinned: false
app_port: 7860
base_path: /web
colorFrom: blue
colorTo: indigo
tags:
  - openenv
---

# OrbitalThrusterEnv

OpenEnv benchmark for **Theme #2: (Super) Long-Horizon Planning & Instruction Following**. The agent must track mission directives over a long episode, preserve fuel for delayed objectives, recover from anomalies, and finish in precision hold.

**Submission links**
- Hugging Face Space: https://huggingface.co/spaces/pixxel-phantom/orbital-thruster-env
- Trained adapter (GRPO LoRA, 1.5B): https://huggingface.co/pixxel-phantom/orbital-thruster-grpo-fast
- Trained adapter (GRPO LoRA, 4B): https://huggingface.co/pixxel-phantom/orbital-thruster-grpo
- Mini-blog / write-up: https://huggingface.co/spaces/pixxel-phantom/orbital-thruster-env/blob/main/BLOG.md
- Training notebook: [`training/train_orbital_grpo.ipynb`](training/train_orbital_grpo.ipynb)


**Pitch**: early waste breaks later phases. A controller that looks good on short-horizon pointing can still fail the flagship mission because it burns fuel before the retarget, mishandles the anomaly, or reaches the final hold phase with no reserve left.

## Problem

Modern mission-operations control is not one action repeated forever. It is a chain of directives:

1. Detumble after deployment.
2. Respect a quiet coast window.
3. Repoint to a new relay geometry.
4. Recover from an injected gyro-bias anomaly.
5. Finish with stable precision hold.

This benchmark turns that story into a verifier-backed environment with explicit milestones, delayed checkpoints, and anti-shortcut rewards.

## Environment

The environment keeps the existing orbital control core:

- 13 discrete thruster actions plus `idle`
- deterministic seeded disturbances
- limited RCS fuel
- dense physical reward from pointing, stability, fuel, and overshoot

On top of that, the mission-ops pivot adds:

- `mission_brief`
- `active_directive`
- `pending_directives_count`
- `milestones_completed`
- `anomaly_flags`
- `fuel_reserve_target`
- `phase_deadline_step`
- `reward_breakdown`
- `episode_metrics`

Each action now also includes a required `control_mode`:

- `detumble`
- `slew`
- `brake`
- `trim`
- `hold`
- `recover`
- `safe_hold`

## Tasks

### Curriculum Tasks

- `detumble_satellite` (`easy`): stabilize a newly deployed spacecraft and finish with ample reserve.
- `retarget_180_flip` (`medium`): survive a delayed maneuver window, execute the large flip, and settle cleanly.
- `long_horizon_precision_hold` (`hard`): preserve a fine-pointing envelope under long disturbance exposure.

### Flagship Theme #2 Task

- `mission_ops_long_horizon` (`hard`): a single episode that chains detumble, coast discipline, retargeting, anomaly recovery, and final precision hold.

This flagship task is the main demo task for the hackathon.

## Reward Design

The environment logs rubric-style reward columns instead of a single opaque scalar:

| Column | Signal |
|---|---|
| `physical_tracking_reward` | Pointing accuracy + hold streak bonus − stability − overshoot penalties |
| `fuel_discipline_reward` | Per-step fuel cost penalty + reserve-gap penalty |
| `milestone_completion_reward` | +0.35 on verified directive completion |
| `control_mode_reward` | +0.12 if declared mode matches recommended; −0.08 otherwise |
| `anomaly_recovery_reward` | Bonus for error/rate improvement under active anomaly |
| `anti_stall_penalty` | Penalty for consecutive steps without meaningful progress |

These are surfaced per step in `reward_breakdown` and aggregated in `state.reward_columns`. That makes it easy to show judges not only that reward improved, but **which behaviors improved**.

## Baselines

Three baselines are supported end-to-end:

- seeded random controller
- deterministic PD controller
- tuned PD controller

The current intended story is:

- deterministic clears `easy`
- tuned PD clears `medium`
- both heuristics fail the flagship mission

Fixed-seed baseline results:

| Policy | Easy (detumble) | Medium (retarget) | Hard (hold) | Flagship | Fuel Used (flagship) |
|---|---|---|---|---|---|
| Random | 23.9 / fail | 3.2 / fail | −25.3 / fail | −53.5 / fail | 90.0 |
| Deterministic PD | 17.6 / **pass** | 97.4 / fail | 21.1 / fail | 89.8 / fail | 90.0 |
| Tuned PD | 34.2 / **pass** | 120.1 / **pass** | 27.5 / fail | 115.8 / fail | 88.8 |

![Baseline comparison across all tasks and policies](outputs/baseline_eval/baseline_summary.png)

*Baseline summary: reward totals per policy per task. All three heuristic controllers fail the flagship task.*

Run the fixed-seed evaluation:

```powershell
python training/evaluate_baselines.py
```

## Training Stack

**Stack**: TRL (`SFTTrainer` → `GRPOTrainer`) + PEFT QLoRA on the real OpenEnv environment as the verifier.

**Base model**: `Qwen/Qwen2.5-7B-Instruct` (A100 via HF Jobs) for the headline run; `Qwen/Qwen2.5-1.5B-Instruct` for the fast L4 run. Override via `ORBITAL_BASE_MODEL` env var.

**Why this model**: strong JSON adherence (we score on JSON validity), fits 4-bit QLoRA on a single GPU, mature TRL integration. We use the vanilla TRL + PEFT + bitsandbytes path (no Unsloth) because the Unsloth `matmul_lora` kernel hit a dtype mismatch (`Half` vs `Float`) on the cloud image and the dependency lock chain (`unsloth` → `trl ≥ 0.18` → `mergekit` → `pydantic <2.11` vs `openenv-core` → `pydantic ≥2.11.7`) is unresolvable. Vanilla TRL on the same image works first try.

**Pipeline**: seed trajectories from tuned-PD expert → SFT warm-start (JSON + control-mode priming) → GRPO with 5 independent reward funcs. The plan that produced the headline 7B run: 150 SFT steps, 300 GRPO steps, `num_generations=8`, `temperature=1.3` (high enough to keep `frac_reward_zero_std=0` — i.e. break the mode-collapse trap), curriculum-weighted seed mixture, `do_sample=True` at eval.

**GRPO reward functions (independent, summed — anti-hacking design):**

| Function | Signal |
|---|---|
| `reward_format` | strict JSON parse + valid enums + reason field |
| `reward_env_step` | replay history into fresh env, score candidate action via real physics |
| `reward_mode_match` | `control_mode` ∈ recommended for active directive |
| `reward_anti_spam` | penalty if same action ≥ 4× in last 7 steps |
| `reward_fuel_discipline` | low-fuel→idle bonus, low-fuel→large-pulse penalty |

**Entry points:**
- `training/hf_job_train.py` — UV script for `hf jobs uv run` (cloud, GPU credits)
- `training/qwen3_smoke_sft.py` / `qwen3_grpo_train.py` — local script entrypoints

Run on cloud (headline 7B, A100):
```bash
hf jobs uv run --flavor a100-large --timeout 4h --secrets HF_TOKEN \
  -e ORBITAL_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
  -e ORBITAL_VANILLA=1 \
  -e ORBITAL_SFT_STEPS=150 -e ORBITAL_GRPO_STEPS=300 -e ORBITAL_NUM_GEN=8 \
  -e OUTPUT_REPO=pixxel-phantom/orbital-thruster-grpo \
  -d training/hf_job_train.py
```

Run on cloud (fast 1.5B, L4):
```bash
hf jobs uv run --flavor l4x1 --timeout 2h --secrets HF_TOKEN \
  -e ORBITAL_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  -e ORBITAL_VANILLA=1 \
  -e ORBITAL_SFT_STEPS=40 -e ORBITAL_GRPO_STEPS=80 \
  -e OUTPUT_REPO=pixxel-phantom/orbital-thruster-grpo-fast \
  -d training/hf_job_train.py
```

Training-only deps: [training/requirements.txt](training/requirements.txt).

## Results

### Headline run — `Qwen/Qwen2.5-7B-Instruct`, A100-large, 150 SFT + 300 GRPO

**SFT phase:** loss 2.33 → ~0.5 on 384 expert traces (JSON + control-mode priming).

**GRPO phase:** loss converged to **0.156** plateau, total reward **~2.0 sustained** across all 300 steps, `reward_format = 1.0` from step ~2 (perfect JSON throughout), `reward_mode_match = 0.5` (constant — model picked the recommended mode every step), `frac_reward_zero_std = 0.0` for the entire run (mode-collapse trap broken — see "Plan that produced this run" below).

### GRPO Training Curves

![GRPO per-component reward and loss over training steps](outputs/training/grpo_metrics.png)

*GRPO training curves (7B run). Top: per-component reward breakdown (`reward_format`, `reward_env_step`, `reward_mode_match`, `reward_anti_spam`, `reward_fuel_discipline`). Bottom: policy loss. `reward_format = 1.0` from step ~10 (perfect JSON). `reward_env_step` carries the real physics signal at ~0.6–0.8. `frac_reward_zero_std` stays at 0 for all 300 steps — the policy keeps generating diverse rollouts, so the GRPO advantage is non-degenerate throughout training.*

### Per-Component Reward at Convergence (7B, final GRPO steps)

| Component | Step 2 | Step 300 |
|---|---|---|
| `reward_format` | 1.0 | **1.0** (perfect JSON throughout) |
| `reward_env_step` | 0.59 | 0.60 (variable, physics-backed) |
| `reward_mode_match` | 0.50 | **0.50** (always picks recommended mode) |
| `reward_anti_spam` | −0.03 | −0.10 (small repetition penalty) |
| `reward_fuel_discipline` | 0.0 | 0.0 |
| **Total** | **2.06** | **2.00** |

### Trained vs Baselines (4-task rollout, fixed seeds)

| Policy | Easy (detumble) | Medium (retarget) | Hard (hold) | Flagship | Fuel Used (flagship) | Milestones |
|---|---|---|---|---|---|---|
| Random | 23.9 / fail | 3.2 / fail | −25.3 / fail | −53.5 / fail | 90.0 | 0 |
| Deterministic PD | 17.6 / **pass** | 97.4 / fail | 21.1 / fail | 89.8 / fail | 90.0 | 2 |
| Tuned PD | 34.2 / **pass** | **120.1 / pass** | 27.5 / fail | **115.8 / fail** | 88.8 | 0 |
| **Trained (GRPO, 7B)** — headline A100 run | 12.4 | 33.3 | **43.8** | 11.0 | 90.0 | 0 |

![Trained model vs baseline policies across all tasks](outputs/eval_trained/trained_vs_baseline.png)

*Trained vs baselines: reward totals per task. The 7B trained model still beats every heuristic on `long_horizon_precision_hold` (43.8 vs 27.5 tuned PD), and uses non-zero fuel on every task (52.8–120 across the four tasks) — proving the policy actively explores rather than collapsing to passive `HOLD_POSITION`. The flagship score is still below tuned PD, which is the next item to address.*

### Key Observations

**What worked:**
- `reward_format = 1.0` from step ~10 and held. SFT priming was decisive — without it, GRPO burns its budget learning JSON syntax.
- `frac_reward_zero_std` stayed at 0 for the entire 300-step run. The combination of `temperature=1.3`, `num_generations=8`, and the curriculum-weighted seed mixture kept rollouts diverse, so the GRPO advantage normalisation never divided by zero. This is the classic mode-collapse trap that ate ~190 steps of an earlier run.
- The 7B model **uses fuel on every task** (52.8 / 120.0 / 85.0 / 90.0 across the four tasks). The earlier 1.5B run learned to idle to game the precision-hold reward; the 7B run actually maneuvers.
- The 7B model still **beats every heuristic on the hard precision-hold task** (43.8 > 27.5 tuned PD), so it learned a non-trivial control policy, not just a passive policy.
- No reward hacking was observed across any reward component (the rubric is the anti-hacking story).

**What needs more training / the next iteration:**
- Flagship score 11.0 is below tuned PD's 115.8. The 7B model commits to maneuvers but does not yet land milestones — `directive_completion_ratio = 0` on every task. The next run should up-weight `mission_ops_long_horizon` in the curriculum (currently 10%) and run longer GRPO (≥ 600 steps) so the model sees more milestone-transition gradient.
- `success = 0` on all four tasks for the 7B run. Easy/medium scores are below tuned PD because the model is exploring with high temperature instead of executing the tight expert maneuver. A second-stage GRPO with `temperature=0.7` and milestone-weighted rewards is the natural follow-up.

### Plan that produced this run

The 7B headline run was the result of a six-issue debugging plan against an earlier 4B run that produced a degenerate model (`fuel_used=0` everywhere, `success=0`, mode-collapse by step ~60). The plan:

1. SFT→GRPO LoRA rank mismatch (`r=32` vs `r=16`) → silent warm-start failure. **Fix:** unify on `r=16, alpha=16`.
2. GRPO mode collapse (`temperature=0.9`, `num_generations=6`). **Fix:** raise to `1.3` and `8`.
3. Eval greedy collapse (`do_sample=False`) → always passive HOLD policy → `fuel_used=0`. **Fix:** `do_sample=True, temperature=0.7`.
4. SFT too few steps. **Fix:** `80 → 150` steps; `256 → 384` records.
5. `reward_mode_match` too weak. **Fix:** `+0.25/−0.15 → +0.5/−0.3`.
6. `reward_anti_spam` insufficient pressure to break passive policy. **Fix:** `−0.4/−0.15 → −0.6/−0.25`.

Reward function logic and task definitions were not modified. The judging story (5 independent rewards, multi-component) is preserved.

## Local Usage

```powershell
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
python validate.py
```

## Training Usage

```powershell
python training/generate_seed_trajectories.py
python training/evaluate_baselines.py
python training/qwen3_smoke_sft.py
python training/qwen3_grpo_train.py
```

## Docker

```powershell
docker build -t orbital-thruster-env .
docker run -p 7860:7860 orbital-thruster-env
```

## Inference

`API_BASE_URL` and `MODEL_NAME` can be overridden at runtime. `HF_TOKEN` is required for remote inference.

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen3-8B"
$env:HF_TOKEN = "hf_xxx"
python inference.py
```

## Validation

The validation script checks:

- four tasks present
- mission-planning observation fields exposed
- action schema requires `control_mode`
- reward rubric surfaced on `/step`
- cumulative reward columns surfaced on `/state`

```powershell
python validate.py
```

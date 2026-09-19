# PLUMB — implementation spec

**Audience: a coding agent with no prior context.** Revised after two rounds of adversarial review, 2026-09-18; implementation evidence updated 2026-09-19. The local control plane and real cluster diagnostics now exist; see §12 and [HANDOFF.md](HANDOFF.md). No Baseten deployment, calibrated judge, or qualified scientific/performance result is established. Unresolved empirical dependencies are explicit gates, not permission to reduce scope.

Build a system that evaluates robot manipulation policies inside a generative video world model on Baseten, scores the rollouts with a two-stage judge, compares results with a published human-scored real-robot reference, and measures the reliability of its own measurements. Preserve per-cell comparisons, supported policy ordering, split-half reliability, minimum detectable difference (MDD), drift, cost-fidelity, judge distillation, reverse validation, and the full live demo.

**Primary matrix: OpenVLA, OpenPiZero, Octo-Small v1.0, MiniVLA, SuSIE, SuSIE_LL × all five tasks × 50 matched starts = 1,500 attempted virtual episodes.** Octo-Base is a separate diagnostic, not a seventh benchmark policy. Pilot runs, calibration, sweeps, confirmation runs, and load rehearsals have separate run IDs and budgets; they are not included in that 1,500.

Evidence labels used throughout: **source-verified** means supported by linked code/docs; **qualification target** means desired but unmeasured; **unresolved** means a dependency still needs evidence. Repository names and historical file sizes are discovery aids, not immutable pins or proof of local downloads. See §12 for the evidence record and missing files.

---

## 0. Qualification gates

Build the minimum adapters, fixture runner, and persistence needed to execute these gates first. The original 30-minute smoke test and one-hour null-test budgets are initial investigation timeboxes, not guarantees that the dependencies can be solved in that time. Every gate starts `not_run`.

| Gate | Evidence required | Blocks |
|---|---|---|
| A — backend conformance | Pinned vendor fixture and Bridge fixture load; correct shapes, frame order, finite values, prompt/domain handling, normalization, actual GPU memory, and warm/cold latency recorded. | Generated evaluation |
| B — action and feedback fidelity | Paired action interventions across tasks; native policy feedback preserved; validated state conversion, forecast-state limitations, supported control horizons, and suffix-causality checks (§2). | Claims about the named policies |
| C — scenario parity | Five task-specific start panels with real image/state provenance, goal references, initial-state distribution and scene/control comparability to AutoEval. | Five-task reference validation |
| D — judge calibration | Blinded human calibration and held-out report, frozen rubric/model/sampling/label mapping, validity and action-text leakage checks. | Primary scoring |
| E — measurement confirmation | Frozen analysis protocol; full matrix; exclusion sensitivity; held-out confirmation of selected cost setting and separately validated distilled judge. | Reliability and cheap-setting claims |
| F — burst qualification | Three complete 1,500-episode rehearsals under the displayed capacity, configuration, timer, and cost definitions (§7). | Public 60-second / ~$11 claim |

Record status (`not_run`, `pass`, `fail`, `blocked`), protocol hash, fixture/start IDs, backend/policy/judge revisions, evidence URIs, measurements, thresholds, and reasons in `results/gates.json`. Define numerical tolerances and effect-size thresholds using independent development fixtures and freeze them before held-out gate evaluation. A changed adapter, backend, judge, or operating point invalidates its dependent gates.

Gate B replaces the single shuffled-action “must fail” rule. Compare original, zero, temporally permuted, sign-reversed, and cross-episode controls at matched starts and seeds. Include already-successful starts and legitimate stationary/contact states as controls. Measure action–video alignment and intervention effect distributions; a shuffled trajectory may still succeed. The success judge never receives actions. Calibrate motion checks against real trajectories instead of equating low optical flow with invalidity.

Keep both Cosmos and IRASim paths. Switching backends requires separate adapters and requalification; IRASim is not a ready drop-in fallback. If a gate fails, retain the requested feature and report its blocker. Diagnostic videos can still be shown as exploratory, without claiming benchmark parity or qualified performance.

---

## 1. Assets, access, and reproducible environments

### World model
Primary: `nvidia/Cosmos3-Nano` (historical download estimate ~33 GB, OpenMDW-1.1). Speed arm: `nvidia/Cosmos3-Edge` (~8.5 GB historical download estimate; its model card lists 4B). Super's model card lists 64B; exclude it from the single-H100 design. Download size, parameter count, and peak inference memory are different quantities. Record measured memory for the selected runtime instead of treating a file size as a fit guarantee. [Nano](https://huggingface.co/nvidia/Cosmos3-Nano), [Edge](https://huggingface.co/nvidia/Cosmos3-Edge), [Super](https://huggingface.co/nvidia/Cosmos3-Super)

Use the official Diffusers `Cosmos3OmniPipeline` forward-dynamics path as the initial production candidate, wrapped in a custom Baseten Chainlet. Pin the model, Diffusers code, domain configuration, CUDA/container, and exact request serializer as one compatibility profile. vLLM-Omni remains an alternative profile; SGLang action support remains unresolved for the chosen version. General video-serving support is not proof of Bridge FD support. Gate A must run the actual deployment payload.

The optional `nvidia/Cosmos-1.0-Guardrail` is gated. The initial fixed robotics-fixture profile uses the documented Diffusers `enable_safety_checker=False` and does not fetch it; record that choice. Do not transfer flags between backends. Free-play uses fixed task prompts and bounded action inputs rather than arbitrary user prompts; any broader deployment needs its own input/content controls.

IRASim: official Bridge archive `https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_checkpoints_data.tar.gz` (~32 GB). Its code is Apache-2.0; inspect archive-specific licenses/notices, hash the archive and extracted weights, and run its own fixtures before marking it usable. [Official IRASim repository](https://github.com/bytedance/IRASim)

### Policy discovery list

These paths and sizes come from the original handoff and source checks. Resolve the exact files at immutable revisions into the asset manifest before download/use; do not claim these bytes are already local.

| Policy | Repo | File | Size |
|---|---|---|---|
| OpenVLA | `openvla/openvla-7b` | all 18 files | 15.085 GB |
| Octo-Small | `rail-berkeley/octo-small` | `270000/default/checkpoint` | 547 MB |
| Octo-Base | `rail-berkeley/octo-base` | `300000/default/checkpoint` | 811 MB |
| MiniVLA | `Stanford-ILIAD/minivla-vq-bridge-prismatic` | `checkpoints/step-362500-epoch-21-loss=0.2259.pt` | 5.55 GB |
| MiniVLA VQ | `Stanford-ILIAD/pretrain_vq` | `pretrain_modvq+mx-bridge_dataset+fach-7+ng-7+nemb-256+nlatent-512/checkpoints/model.pt` | 9.5 MB |
| Open pi-zero | `allenzren/open-pi-zero` | `bridge_beta_step19296_2024-12-26_22-30_42.pt` | 11.773 GB |

**Traps:**
- Octo checkpoints are NOT at repo root. `resolve/main/checkpoint` returns 404.
- Octo must be v1.0. Do **not** use `octo-small-1.5`.
- MiniVLA's filename contains `=` — URL-encode as `%3D`. Skip the 76 MB `.jsonl`, it is a training log.
- Continuation source audit: MiniVLA's VQ configuration models seven future actions, but the pinned released tokenizer returns only `ret_action[:, 0]`. An exposed seven-action proposal is therefore still unresolved, not implemented by repeating that action. See [the dependency record](docs/POLICY-DEPENDENCIES.md) for pinned code and loader blockers.
- `Stanford-ILIAD/pretrain_vq` declares **no license** (`cardData: null`). Flag to the user; do not redistribute.
- Remove automatic mirroring. Resolve missing license/access terms for `pretrain_vq`; private hosting is not a substitute for redistribution rights. Preserve upstream notices for any authorized archival copy. MiniVLA remains required, with license resolution tracked as a dependency.

Open pi-zero's checkpoint is intended to supply the model weights; verify full state-dict coverage in the pinned loader. `strict=True` checks keys, not provenance or deserialization safety. Required PaliGemma support files are `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, `added_tokens.json`, `special_tokens_map.json`, `preprocessor_config.json`, and `config.json` (historical estimate 21.9 MB). Prefer the official `google/paligemma-3b-pt-224` access path under accepted terms. `leo009/paligemma-3b-pt-224` is a candidate mirror only after provenance, hashes, and applicable terms are resolved; do not claim byte identity without a saved comparison. Configure an explicit application cache directory. [Official PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224)

SuSIE and SuSIE_LL are **required for the full matrix**. Candidate assets: `patreya/gcbc-bridge` (`checkpoint_75000`, historical size 258,718,956 B; advertised MIT) and `kvablack/susie` (~3.438 GB). SuSIE needs separate subgoal and low-level components plus a pinned JAX/Flax Stable Diffusion stack. SuSIE_LL executes the low-level goal-conditioned policy directly. Preserve the released AutoEval configuration as the replication arm, disclose its mismatch with upstream SuSIE, and evaluate a corrected upstream configuration as a separately named sensitivity arm (§4).

### Judge and data

- Rubric-judge candidate: `Qwen/Qwen2.5-VL-7B-Instruct`, served on Baseten with a pinned runtime/processor. It supports image/video inputs; robot-scoring quality still requires Gate D. [Model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Retain `facebook/vjepa2-vitl-fpc64-256` as a separate frozen video-feature diagnostic. It is an encoder, not a text-following rubric judge. Use its documented 64-frame preprocessing and explicit timestamp sampling, outside the deterministic validity gate. Candidate files: `config.json` (785 B), `model.safetensors` (1,303,947,864 B), `video_preprocessor_config.json` (1,298 B); skip duplicate `original/model.pth`. [Model card](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)
- `IPEC-COMMUNITY/bridge_orig_lerobot` — dataset card advertises Apache-2.0 and 99,673 files. Pull only manifest-selected episodes. It is not automatically the same scene distribution as AutoEval. [Dataset](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot)
- `zhouzypaul/auto_eval` — original handoff estimates 173.6 GB and 12,054 episodes, with video, 7-D actions, 8-D proprio, and classifier labels. The public index exposes the drawer scene; it does not establish matched sink/cloth panels or policy identity. Use it for drawer validation and human reannotation, not as human-labelled five-task ground truth. Some pickle objects refer to `robot_eval_logger` and `wandb`; a stub unpickler is not a security boundary. [Dataset](https://huggingface.co/datasets/zhouzypaul/auto_eval)

### Starting environment constraints — isolate per component, then lock

The following are compatibility leads from the policy handoff, not a fully tested environment. Resolve exact transitive versions, CUDA/JAX wheels and code commits inside separate policy images, then save container digests. The world model and judge need their own modern runtime images.
```
# OpenVLA
timm==0.9.10 · tokenizers==0.19.1 · torch>=2.2.0 · torchvision>=0.16.0 · transformers==4.40.1
# load: AutoModelForVision2Seq.from_pretrained(..., trust_remote_code=True), unnorm_key="bridge_orig"

# Octo (both)
python 3.10 · jax[cuda11_pip]==0.4.20
# obs_horizon=2, pred_horizon=4, temporal ensembling ON

# MiniVLA
python 3.10 · torch 2.2.0 · torchvision 0.17.0 · transformers 4.40.1 · tokenizers 0.19.1 · timm 0.9.10 · flash-attn 2.5.5
# repo: Stanford-ILIAD/openvla-mini · use_extra=True · action chunk 7

# Open pi-zero
# repo: allenzren/open-pi-zero · uv sync · action chunk 4 · action_normalization_type="bounds"
```

The AutoEval wrapper's `bounds` versus `normal` setting is adapter-specific, not a universal native-policy API. Verify each wrapper's training statistics, action denormalization, gripper transformation, and clipping against pinned source and a golden action fixture.

### Asset manifest and loading

Before Gate A, create `assets.lock.json`: repo ID/type, immutable commit, file path, byte length, SHA-256, source URL, retrieval time, license/access status and notices, loader/code revision, container digest, and compatibility-profile ID. Use actual downloaded hashes; unknown values stay null and block a `verified` status. Tokenizer/model combinations and action normalizers are part of the lock.

Pin and review remote code before enabling `trust_remote_code`. Construct known architectures and prefer safetensors or supported weights-only loaders. If legacy `.pt`/`.pkl` conversion is necessary, isolate it in a disposable unprivileged environment without credentials or network, validate output schemas, and retain converted artifact hashes. A pickle scan or `strict=True` alone does not establish safety. Normal inference consumes the reviewed converted artifacts. [PyTorch loading guidance](https://docs.pytorch.org/docs/stable/generated/torch.load.html), [Hugging Face pickle guidance](https://huggingface.co/docs/hub/security-pickle)

---

## 2. Policy, control, and world interfaces

### Canonical observations and policy adapters

Use a `ControlTrajectory` with timestamped Bridge control ticks, absolute commanded pose, physical 7-D actions `[dx, dy, dz, droll, dpitch, dyaw, gripper]`, source-state lineage, and generated observations. Initial image format is 256×256 RGB uint8; model-specific resizing belongs in the adapter. The initial Bridge state convention is `(x, y, z, rx, ry, rz, 0, gripper)`; the handoff's 0–0.39 gripper-state range and closed-at-zero convention must pass a source-data fixture. State units and action gripper units are not interchangeable.

Each `PolicyAdapter` declares required observation history, proprio fields, native proposal horizon, execute/replan rule, temporal ensembling, preprocessing, normalization, reset behavior, and RNG stream. Starting contracts to verify against the released AutoEval wrappers:

| Policy | Native contract to certify |
|---|---|
| OpenVLA | One 7-D action per fresh image; no proprio input in its base Bridge API. |
| Octo-Small / Octo-Base | Two-image history and four-action proposal; verify temporal ensembling and executed prefix against the wrapper. |
| MiniVLA | Seven-action proposal in the supplied configuration; verify executed prefix and history. |
| OpenPiZero | Four-action proposal and refreshed proprioception. |
| SuSIE / SuSIE_LL | Goal-image inputs, subgoal cadence where applicable, and low-level execution rule from the released wrapper; unresolved until fixture-tested. |

Never fabricate a 16-row proposal by repeatedly querying a single-step policy against the same image. Execution horizon belongs to the evaluated policy wrapper, not the video transport. Preserve the benchmark wrapper for the replication arm; changes receive a new policy/protocol identity. [OpenVLA control loop](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/bridge/run_bridgev2_eval.py#L245-L280), [AutoEval policy wrappers](https://github.com/zhouzypaul/auto_eval/blob/3ea3ff44c6950433cfbcb4294a3deaa616533745/auto_eval/robot/policy.py)

Video generation does not supply measured robot state. `BridgeControlIntegrator` must reproduce the pinned Bridge controller's local-action transform, gripper semantics, clipping, and target-pose update before `BridgeToCosmosCompiler`; simple Euler/position addition is not an assumed equivalent. Record whether integration starts from measured EEF pose or the last commanded pose. Its output is `forecast_proprio`, with errors characterized against held-out real trajectories. Compare forecast versus frozen-state ablations. Neither is a physical state measurement; never label forecast-state rollouts equivalent to real closed-loop control without supporting evidence. Policies that do not consume proprio still need a pose estimate for the Cosmos action compiler. [Pinned Bridge controller](https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_envs/base/robot_base_env.py#L182-L246)

Freeze one image-feedback and state-feedback mode per policy/backend/cell; never pool modes. Qualified reference-comparison cells require Gate-B-qualified generated-image feedback and a declared state mode. Forecast/frozen-state or padded-prefix results stay in explicitly named approximation/sensitivity cohorts unless their preregistered equivalence tolerances pass; even then retain the mode label and limitation. An unresolved mode blocks that qualified cell rather than being silently counted as a faithful replica.

### Backend action compilation

`BridgeToCosmosCompiler(control_trajectory, state, normalizer_revision)` constructs consecutive absolute Bridge poses, applies the pinned state→FK, TCP→flange, and Bridge→OpenCV transforms, then computes the backward-framewise local relative transform `T_i^-1 @ T_(i+1)`. Encode translation, the rotation matrix's first two columns in rot6d order, and gripper as 10D. Apply the selected checkpoint's normalization exactly once, at its documented boundary. A standalone Euler→rot6d round trip cannot validate this conversion, and “raw metres, no scaling” is not established for every checkpoint. Use golden real-state/action fixtures, nonidentity rotations, gripper endpoints, normalizer inversion, and matrix-equivalence tests. [Pinned Bridge implementation](https://github.com/NVIDIA/cosmos-framework/blob/c23e51f2f157ae3e51cfcd86ebfb5464850894f2/cosmos_framework/data/generator/action/datasets/bridge_orig_lerobot_dataset.py), [pose conventions](https://github.com/NVIDIA/cosmos-framework/blob/c23e51f2f157ae3e51cfcd86ebfb5464850894f2/cosmos_framework/data/generator/action/utils/pose_utils.py)

`IRASimBridgeAdapter` has a different native contract: 256×320 images, 16 frames including the condition frame, 15 7-D action rows, and scaling `[20,20,20,20,20,20,1]` in the official Bridge data path. Verify frame convention and action alignment from its own fixtures. Do not pass Cosmos 10-D actions into IRASim. Keep backend-specific metrics separate. [IRASim config](https://github.com/bytedance/IRASim/blob/c72b6dade6fcd65971e0aa8ab49ea39b15108c90/configs/evaluation/bridge/frame_ada.yaml), [data adapter](https://github.com/bytedance/IRASim/blob/c72b6dade6fcd65971e0aa8ab49ea39b15108c90/dataset/dataset_3D.py)

### WorldAdapter contract and feedback gate

`WorldRequest` contains the conditioning image, exact task prompt, domain, compiled actions, nominal control timestamps, generation seed, and compatibility-profile ID. `WorldResult` contains frames with wrapper-assigned nominal timestamps from the pinned control/frame mapping, conditioning-frame metadata, server timing, and artifact hashes. Only real-robot source records may be labelled measured physical control timestamps. Bind Bridge's domain name/dimension to the pinned registry, not an unversioned numeric ID.

The Cosmos example uses one condition frame, 16 actions, and 17 returned frames. Remove the repeated condition frame when stitching; retain the original start frame separately and score through the exact task control horizon. Keep control rate, generated-frame rate, and presentation playback rate explicit; the intended 5 Hz control mapping requires fixture verification and is not automatically physical blocking-control timing.

Request fields differ: the official vLLM-Omni example includes `num_frames=action_chunk_size+1`; Framework FD uses `vision_path`/`action_path` plus prompt and domain. The reviewed source does not establish a universal Bridge multiple-of-four restriction. Do not transfer video-to-video conditioning arguments into action mode. Probe supported action lengths, output timestamps, conditioning semantics, and payloads on the pinned deployment. [Nano FD example](https://huggingface.co/nvidia/Cosmos3-Nano/blob/e59a53c25979a090fa8706c9acc0c254a6e89b92/README.md), [Framework FD contract](https://github.com/NVIDIA/cosmos-framework/blob/c23e51f2f157ae3e51cfcd86ebfb5464850894f2/docs/inference.md)

**Unresolved blocking capability:** OpenVLA needs a fresh generated observation after every executed action, but a 16-action diffusion request commits future actions before those observations exist. First test native one-tick FD support and action grounding. If only longer chunks work, test changing future action suffixes while fixing the consumed prefix and seed. Generating a long chunk and consuming its first frame is an approximation even if it passes this test; record its protocol identity and residual sensitivity. A failed test blocks the named-policy fidelity claim. It is not resolved by action repetition, stale frames, or silently dropping OpenVLA.

Only for a policy/backend pair that passes Gate B, the qualified controller loops by the adapter's **certified executed prefix**, queries the policy at its specified feedback boundaries, compiles actions, calls the world adapter with the task prompt, updates history and forecast state, and persists events. Otherwise it may emit separately labelled exploratory approximations, but cannot populate a qualified named-policy cell. It never assumes five chunks. For terminal padding, demonstrate prefix invariance to post-horizon actions; otherwise require an exact supported terminal length before claiming matched task horizons.

---

## 3. Tasks — verbatim, do not paraphrase

| # | Prompt (exact string) | Success | Max steps |
|---|---|---|---|
| 1 | `Close the drawer` | completely closed | 70 |
| 2 | `Open the drawer` | opened >= 1.5 cm | 70 |
| 3 | `Put the eggplant in the yellow basket` | fully inside at episode end | 100 |
| 4 | `Put the eggplant in the blue sink` | fully inside at episode end | 100 |
| 5 | `fold the cloth from top right to bottom left` | folded >= 1/4 diagonally | 80 |

Note the lowercase `fold` in task 5. Reproduce it.

Scene parity must include the original handoff's fixture details: drawer and cloth attachment and plastic wrap over the sink. Verify these against source scene records before treating them as facts about a selected frame. Success criteria must be visible or measurable; a 1.5 cm drawer threshold needs a scene-specific calibrated reference, not an unsupported metric estimate from pixels.

### ScenarioManifest and splits

For each task, provide 50 distinct real starting states, shared across all six policies, plus disjoint development/calibration and cost-confirmation panels. A start record includes `start_id`, source dataset/revision/episode/frame and hashes, `start_lineage_id`, image/state timestamps, state convention, camera/crop/calibration, scene/object configuration, initial-state stratum, exact instruction, and goal-reference hashes/provenance. Neighboring frames or repeated world seeds from one physical start are not independent new starts.

Gate C compares scene layout, objects, reset/randomization distribution, policy checkpoint/wrapper, control timing and task horizon with AutoEval. The published aggregate tables do not supply the original trial-level starts. Unless exact source states are recovered, describe a matched-distribution comparison with documented limitations, not paired reconstruction of the original trials. Lack of sink/cloth assets remains an explicit acquisition dependency; do not fill these cells with drawer starts.

---

## 4. Published reference — preserve these tables

Human-scored, 50 trials per cell. Source: [AutoEval, Table 2](https://arxiv.org/html/2503.24278v2). These are the paper's real-robot results, not PLUMB outcomes. Preserve its `Octo` table key and explicitly map it to Octo-Small in the policy manifest. Use only the common four-task subset when comparing against SIMPLER; never insert a zero for its absent cloth task.

```python
HUMAN = {  # successes out of 50
  "OpenVLA":     {"open_drawer":40, "close_drawer":46, "to_basket": 1, "to_sink": 0, "fold_cloth":12},
  "OpenPiZero":  {"open_drawer":24, "close_drawer":45, "to_basket": 7, "to_sink":47, "fold_cloth": 3},
  "Octo":        {"open_drawer": 0, "close_drawer": 0, "to_basket": 0, "to_sink": 0, "fold_cloth": 2},
  "MiniVLA":     {"open_drawer":32, "close_drawer":49, "to_basket":38, "to_sink": 0, "fold_cloth": 8},
  "SuSIE":       {"open_drawer": 2, "close_drawer":13, "to_basket": 0, "to_sink": 0, "fold_cloth":10},
  "SuSIE_LL":    {"open_drawer": 0, "close_drawer": 0, "to_basket": 0, "to_sink": 0, "fold_cloth": 0},
}

SIMPLER = {  # physics simulator, same paper Table 3. NO fold_cloth column.
  "OpenVLA":     {"open_drawer":32, "close_drawer": 2, "to_basket": 1, "to_sink": 0},
  "OpenPiZero":  {"open_drawer":34, "close_drawer":24, "to_basket":45, "to_sink": 6},
  "Octo":        {"open_drawer": 3, "close_drawer": 0, "to_basket": 6, "to_sink": 3},
  "MiniVLA":     {"open_drawer":30, "close_drawer":23, "to_basket":10, "to_sink": 2},
  "SuSIE":       {"open_drawer": 0, "close_drawer":41, "to_basket": 7, "to_sink": 0},
  "SuSIE_LL":    {"open_drawer": 1, "close_drawer": 0, "to_basket": 0, "to_sink": 0},
}
```

**The headline cell: OpenVLA / close_drawer — 46/50 = 92% human vs 2/50 = 4% simulated. An 88-point error.** Next largest: OpenPiZero/to_sink 94% -> 12%, OpenPiZero/to_basket 14% -> 90%.

**Report per-cell virtual rates, reference discrepancies, and supported orderings with uncertainty.** MiniVLA totals 127/250 and OpenPiZero 126/250. Removing open, close, basket, or cloth reverses that pair; removing sink leaves MiniVLA at 127 versus OpenPiZero at 79. The one-trial aggregate lead is not evidence of a resolved ranking. Do not force a total order or claim tier membership from an arbitrary clustering rule.

Two sensitivity analyses belong in the report:

1. Octo totals **2/250**, including two cloth successes. Its low performance is a reason to audit wrapper/scene differences and report agreement with and without Octo, not proof of a bug. Cross-paper rates from different scenes/protocols are contextual evidence, not interchangeable controls.
2. AutoEval's released SuSIE configuration uses `gc_bc`, while upstream SuSIE specifies `gc_ddpm_bc`. Disclose this configuration mismatch and distinguish the released-config replication arm from a corrected upstream sensitivity arm. Do not replace the primary cell silently. [Released config](https://github.com/zhouzypaul/auto_eval/blob/3ea3ff44c6950433cfbcb4294a3deaa616533745/scripts/configs/eval_config.py), [upstream SuSIE](https://github.com/kvablack/susie)

---

## 5. The judge — two stages, frozen before primary scoring

### Stage A: deterministic validity checks

Python checks frame count/order, decode integrity, finite action/state values, impossible discontinuities and calibrated command–motion consistency. Use classical tracking/flow with fixed parameters and camera/scene references. Whole-image optical-flow magnitude cannot by itself identify arm motion, contact failure, or physical collisions. A stationary gripper pressing a drawer can be valid. Ambiguous observations are `unknown`, not automatically invalid. V-JEPA feature diagnostics remain a separate analysis and cannot silently alter this deterministic gate.

Record `validity = valid | invalid | unknown` and reason codes. Judge-level uncertainty can also make an outcome unevaluable. Do not silently delete such episodes or count them as known physical failures. Keep all attempts in coverage and partial-identification reporting (§6).

### Stage B: blinded VLM rubric

Initial candidate: Qwen2.5-VL-7B-Instruct on Baseten, with processor, weights, runtime, prompt and schema pinned. Input is exactly 16 timestamped frames sampled uniformly from the original start through the exact final control tick (including both endpoints), the task instruction, task-specific rubric, and provenance-backed goal/reference images. Reject malformed/too-short clips rather than silently altering sampling. No policy name, action text, commands, reference success percentages, condition labels or gate outcome reach this judge. Frame timestamps are control timestamps, not display playback time.

Require structured output: `integrity` (`intact`, `artifact`, `uncertain`), `collision` (`none_visible`, `visible`, `uncertain`), `progress` (integer 0–5 or null), `completion_evidence` (`met`, `not_met`, `uncertain`), evidence frame indices, and concise observable reasons. Validate JSON and enums in Python. Collision is reported separately; it does not automatically override a task's published success rule. Artifact/uncertainty is separate from the progress scale.

Task-specific 0–5 milestone definitions are frozen in the rubric:

| Task family | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Drawer | No directed approach | Approach handle | Handle/contact established | Drawer moves toward target | Partial target opening/closure | Published final-state criterion met |
| Pick/place | No directed approach | Approach eggplant | Grasp/contact established | Lifted | Transported over destination | Fully inside destination at end |
| Cloth | No directed approach | Approach designated corner | Corner grasp/contact | Corner lifted | Moved diagonally toward target | Published fold criterion met |

Score final completion from visible end state; an initially satisfied state can be complete without visiting every milestone. For incomplete episodes, record the highest clearly observed milestone; do not infer invisible events. An episode can therefore have progress 4 and binary failure, but progress 5 requires final completion. Unknown final-state evidence makes `binary_success` null.

The initial primary judge protocol uses five independent samples at temperature 0.7, top-p 1.0, with logged seeds and a fixed output-token cap of 512. This is a proposed operating point, not a claim of optimality. Greedy inference is a development-only diagnostic under this protocol. Development can reject/revise the primary protocol before held-out calibration; any revision must explicitly freeze its sample count and corresponding quorum, schema and cost model together. The current protocol's count stays five and quorum stays three. Include every sample/retry in cost. Nonzero temperature is not a universal API requirement or proven accuracy rule.

Aggregation: a sample is decisive only if its schema is valid, integrity is `intact`, progress is numeric, and completion evidence is definite. At least three of five samples must agree on completion for an episode-level decision; otherwise it is unevaluable. Use the lower median progress of the agreeing decisive samples so an even-sized consensus remains an integer observable milestone. Enforce progress 5 for `met` and 0–4 for `not_met`; inconsistent samples are invalid outputs, never repaired into successes. Stage A invalid/unknown episodes remain unevaluable for primary rates. Retain all raw votes, progress, collision and validity fields for sensitivity analysis.

One bounded retry per sample is allowed for transport/schema failure, using the same sampling configuration; record both attempts and the reason. Refusals, exhausted retries, and disagreement are explicit missing outcomes, not forced labels. Five samples are not five independent robot episodes.

### Calibration and distillation

Use **150 generated clips: 100 development and 50 held-out**, stratified across all five tasks (20 development and 10 held-out per task) and covering policies, apparent successes/failures, and generation defects. Freeze selection before annotating. Each development clip receives one annotation; both annotators independently label all 50 held-out clips, giving the original 50-clip overlap and 200 total ratings. Keep source-state lineages disjoint from the primary study and cost confirmation.

Annotators are blinded to policy, backend, commands, source condition and model label. Compute human-human raw agreement and binary/ordinal kappa on their independent labels. For each field, identical labels define consensus; disagreement remains unresolved under this two-annotator protocol and stays in calibration coverage/missingness counts. No project-team tie-break after seeing VLM output is permitted. Report judge confusion matrices, sensitivity/specificity, weighted kappa for progress, uncertainty, and leniency offset (judge-positive minus human-positive proportion on the same consensus-labelled cases). Human-human agreement is a reproducibility reference, not a mathematical ceiling. Stratification must be disclosed; prevalence-sensitive statistics from an enriched calibration sample are not population estimates without sampling weights.

Choose rubric and sampling settings only on development data. Freeze the judge before evaluating the 50 held-out clips and before primary scoring. Gate D's acceptance target is pooled across the five tasks; each task's 10-clip result is exploratory, with intervals. Passing the pooled gate does not establish tight task-specific accuracy. If a later claim requires task-specific qualification, collect an additional independent task-stratified panel before making that claim. Define the acceptable error/coverage tolerances and minimum consensus-labelled coverage in the preregistration before exposing held-out labels. If the gate fails, revise and obtain a fresh held-out set; do not tune against and reuse the same test set as untouched evidence.

Distill on Baseten Training Jobs as a separate judge revision. Keep development/training, held-out calibration and primary evaluation lineages disjoint. LoRA `lr=1e-3, r=64, alpha=32` is one candidate configuration, not a portable optimum; compare a preregistered small search with early stopping on development validation. Training labels/compute are an additional acquisition/budget item, not supplied by the 50 held-out clips. A distilled judge must pass fresh held-out calibration and paired frozen-video comparison before scoring a burst; preserve the original judge's results.

---

## 6. Measurement protocol

Freeze the scenario/policy matrix, split membership, RNG seeds, judge, endpoints, thresholds, exclusion policy, cost-selection rule and analysis code before primary evaluation. Development fixtures may precede this freeze. Store `protocol.json`, its SHA-256 and the analysis-code revision in an append-only project-owned remote artifact store with an externally auditable timestamp and reviewer access; record its URI, version and timestamp in `results/gates.json`. A signed remote release is an acceptable equivalent. Amendments receive a new protocol ID without overwriting the old one. A private source repository now exists, but no qualified primary-study preregistration has been published; a local hash alone does not supply independent timing evidence. Do not call the already-known published 92% value a blind target.

### Rates, missing outcomes, and reference uncertainty

Each cell has 50 planned logical episodes. Record submission/attempt/completion stages separately; terminal service failures and unavailable outcomes remain in the primary intent-to-evaluate denominator. A never-executed cell is incomplete, not a zero-success measurement. Let `N` be all planned episodes for a completed run ledger, `V` those with evaluable binary outcomes, and `S` evaluable successes. Publish:

- Counts, coverage `V/N`, conditional virtual success `S/V`, and all-attempt observed-positive rate `S/N` (a lower-bound convention, not known physical success).
- No-assumption Horowitz–Manski missing-outcome bounds `[S/N, (S + N - V)/N]`. Their point-bound width is exactly the missing-outcome rate; a sampling confidence region need not have that width. Call it missingness when it includes service failures, not just world-model exclusions.
- Reason-specific invalid/unknown/judge/service counts. Keep `S/V` null when `V=0`; the all-missing interval is `[0,1]`.
- Wilson 95% intervals for independent-start binary cell rates and the published human cells. For repeated seeds or correlated starts, retain descriptive Wilson values only with that caveat and use lineage-block uncertainty for inference.

Resample start-state bundles **within each fixed task**, carrying every policy's outcomes for a selected start together; use 10,000 deterministic-seed bootstrap replicates. Cluster shared source/reset lineages together. The five benchmark tasks are not a random sample of all robotics tasks: do not bootstrap only five tasks and claim task-population generalization. Equal-weight tasks for macro rates, with leave-one-task-out sensitivity reported separately.

Propagate the reference's finite `n=50` per cell separately using a disclosed binomial-model parametric bootstrap; trial-level pairing/correlation in the original study is unavailable. Report conditional-on-reference and reference-uncertainty sensitivity results distinctly. Never pair generated episodes with human trials whose identities are unknown.

Define the primary observed-positive endpoint `Y=1{evaluable AND success}` for every planned start; missing outcomes have `Y=0` only for this explicitly labelled lower-bound endpoint. For each paired cell contrast `mean(Y_A-Y_B)`, report a two-sided exact McNemar test and paired-start bootstrap interval. Because the report displays all tasks, use Bonferroni alpha `0.05/75` across 15 pairs × five tasks; macro comparisons form a separately identified 15-pair family at `0.05/15`, not a joint guarantee across both report families. Complete-pair conditional-success contrasts are secondary, selection-sensitive analyses with their pair count and coverage disclosed.

A significant observed-positive contrast is not by itself evidence of unconditional virtual-success ordering. In each bootstrap replicate compute `L=S/N`, `U=(S+N-V)/N` and the conservative contrast `L_A-U_B`; support A over B only when its one-sided bootstrap lower confidence bound at alpha `0.05/75` for task comparisons (or `0.05/15` for the separately reported macro family) is positive. Guard against degenerate bootstrap certainty at zero/one: for independent-start primary cells also require separation of simultaneous exact binomial endpoint envelopes, using one-sided tail `0.05/60` for each of the 30 cells' lower/upper endpoints. Calculate these from the binary indicators `Y` and `Z=1{success OR missing}` respectively. Macro endpoint envelopes are equal-task averages of these cell envelopes. For clustered-start designs lacking a validated boundary-aware confidence method, leave the ordering indeterminate rather than claim nominal coverage. Scoreboards may sort estimates visually but must retain this indeterminacy rule; no arbitrary fixed tiers or strict six-policy permutation.

### The four headline measurements

1. **Split-half reliability:** repeat 1,000 balanced 25/25 splits of start IDs within each task; carry all six policies together. Correlate the 30-cell score vectors, report rate disagreement and pairwise-order stability as well as the correlation distribution. For correlated source starts, split whole lineages and report actual half sizes. All-attempt positive rates define the primary repeatability vector; complete-case results and coverage are sensitivities. Constant vectors yield `undefined`, not perfect correlation. Each split is disjoint; repeated splits may reuse episodes across repetitions and are not 1,000 independent studies. Use an outer lineage bootstrap for uncertainty rather than interpreting split quantiles as a confidence interval. Repeatability does not establish validity.
2. **MDD:** report power curves for the named observed-positive paired cell test at `n=50` (exact McNemar, alpha `0.05/75`) and the fixed-five-task macro endpoint (equal-task mean paired difference, task-stratified within-start label-swap test, alpha `0.05/15`). The macro randomization null assumes policy-label exchangeability within matched starts; state that assumption. With the primary design's equal 50-start task sizes, obtain its exact sign-swap null from the binomial distribution over discordant pairs, avoiding nested Monte Carlo tests. Power simulations hold designated policy A's per-task marginal fixed, increase the comparison policy's marginal by the tested feasible gap, and vary discordance within joint-probability constraints; save the exact alternative table at each gap. Use at least 10,000 simulations per 1-percentage-point gap and report the power range across the frozen feasible/estimated discordance scenarios. The conservative MDD is the smallest feasible grid gap whose minimum scenario power reaches 80%; also report scenario-specific MDDs and mark a target unattainable on the tested grid when none passes. Distinguish development-based planning curves from retrospective curves whose nuisance/joint probabilities come from the completed primary matrix; the three-policy pilot cannot estimate all six-policy/five-task pairs. Report baseline rate, coverage, `n`, assumptions and Monte Carlo uncertainty. Show sensitivity to missingness, and do not call an MDD for `Y` an unconditional real-success MDD. A universal observed-variance number or a one-trial rank gap is not an MDD.
3. **Drift:** the x-axis is control ticks/nominal elapsed time. Preserve the 1–6-request display only with its backend, action length and feedback cadence stated; it is backend-specific transport/horizon sensitivity, not a common physical duration across policies. Record `(control ticks, FD-request action length, autoregressive boundaries, policy feedback boundaries)`. Separately compare teacher-forced versus free-running predictions on held-out real action/video sequences at equal control prefixes, using the same actions and generation settings. Measure action alignment, integrity and visual divergence. A fixed-horizon partition sweep tests sensitivity to chunking, not isolated drift if chunk length also changes. Report the last tested horizon meeting preregistered tolerances; “no failure observed through H” is not an unlimited drift guarantee.
4. **Cost-fidelity:** vary validated resolution, diffusion steps and chunk partitions with the same task caps, starts, seeds and judge. Lower resolution or fewer steps may reduce quality; a shorter task is not a cheaper equivalent evaluation. Select the cheapest setting satisfying preregistered per-cell error, coverage and pairwise-order tolerances on development sweeps, then confirm on a disjoint panel. Preserve failed settings and all costs; a single noisy ranking match does not prove fidelity.

Also report binary and ordinal-progress agreement with the reference (ordinal comparison is association with human binary rates, not absolute calibration). Pin MMRV to [SIMPLER's implementation](https://github.com/simpler-env/SimplerEnv/blob/06accaca93535902d408da4855f21cece12bceb7/simpler_env/utils/metrics.py#L148-L163): arguments `(perf_sim, perf_real)`, strict order violations, and real-performance margin weighting. Unit-test ties and argument order against hand-worked cases. Primary macro MMRV uses the all-attempt observed-positive/lower-bound virtual rate `S/N`: `MMRV(mean_task (S/N), mean_task human_rate)`. Complete-case MMRV instead uses `S/V` and is conditional; separately report all per-task MMRVs and their mean, which is a different statistic. Complete-case MMRV is explicitly conditional and undefined when a required cell has no evaluable outcomes. Recompute both statistics under the four deterministic missing-outcome completion scenarios: all missing fail; all missing succeed; missing outcomes favor policies higher in the human aggregate ordering; and the reverse. In the last two scenarios, the top three human-aggregate policies receive success completions and bottom three failure completions, or vice versa; resolve human-score ties by the frozen policy-ID order. The all-missing-fail scenario reproduces the primary `S/N` endpoint as a consistency check. These scenarios are sensitivity checks, not exhaustive extrema. The random-order baseline enumerates all `6! = 720` permutations of the relevant six-policy vector at macro level and separately per task; report the mean of per-task expected baselines for the task-summary statistic, retaining ties and the fixed human vector. Report every statistic's effective `n`, lineage count, exclusions and protocol revision; avoid Fisher-transform normal approximations for tiny policy sets.

Assert no source lineage crosses development, held-out calibration, primary and cost-confirmation cohorts; no lineage crosses the halves of a particular reliability split. Pairing across policies or sweep arms is intentional and must not be mistaken for independent replication.

---

## 7. Baseten execution, telemetry, and economics

### Chain topology and durable completion

Use a CPU `RolloutController` Chain entrypoint with independently deployed policy, world-model, validity and judge Chainlets. Policies with incompatible dependency stacks use separate images. The controller performs sequential policy/world feedback for each episode; parallelism comes from independent episodes. Begin world-model benchmarking on one H100 with `concurrency_target=1` and matching in-container concurrency. Choose final hardware/batching/concurrency by measured memory and full-chain throughput. Record these values explicitly; there is no assumed universal “throughput preset.”

Submit complete logical episodes through the deployment-provided Chain **`/async_run_remote`** URL. `/async_predict` is for model deployments. Only the Chain entrypoint uses the external async queue; internal Chainlet requests are awaited RPCs. [Chain invocation](https://docs.baseten.co/development/chain/invocation), [async inference](https://docs.baseten.co/inference/async)

The application control plane owns a durable run ledger, outbox and object storage. Before submission, create one logical row per `(run_id, policy_variant, task, start_id, world_seed, protocol_hash)`; persist the returned platform request ID. Transport retries create attempts, not new statistical episodes. Use atomic episode leases/compare-and-set transitions and stable chunk event IDs; duplicate submissions/webhooks must not double-score, reset seeds or increment the completed counter twice.

Persist each completed world segment, actions, frame/state hashes, policy history, RNG state and timings before advancing. Resume only from a committed boundary with the same inputs; an uncertain in-flight call may repeat compute, whose cost still counts. Write the final artifact before marking the logical episode terminal. Authenticate callbacks using the chosen platform-supported mechanism and validate their run/request association. A reconciler detects missed callbacks, unfinished attempts and timeouts. Queue status is not an output store, and Baseten warns that failed webhook delivery can lose outputs; therefore the Chain persists results independently of callback delivery.

Bound service retries and per-run deadlines in the frozen run configuration (initial default: two transport retries per segment and a 30-minute job deadline; the demo's 60-second qualification timer is separate). Retry only eligible transient failures; do not repeatedly rerun visually invalid episodes until they succeed. Retain every attempt and expense. Cancellation leaves explicit terminal records and accounts for work already allocated.

### Telemetry contract

| Display | Source and meaning |
|---|---|
| Planned, submitted, completed, failed, cancelled, evaluable, excluded | Application ledger; logical episodes, not request-attempt counts |
| Queued / in progress | Baseten `async_queue_status` only after testing the exact Chain deployment route/schema; the cited model-scoped API does not establish Chain routing. Until verified, show platform queue unavailable. Request counts can differ from logical counts during retries. |
| Active replicas per Chainlet | Chain deployment management API |
| Desired / starting replicas | Supported metrics export; show unavailable if the account does not expose it, never fabricate from max replicas |
| Segment/video updates | Persisted `segment_completed` events delivered through application SSE/WebSocket |
| Compute seconds / allocated GPU-seconds | Instrumented per-stage timings / separate resource-allocation ledger; label separately |
| Estimated USD | Allocation ledger × timestamped account resource prices, plus applicable storage/network/service charges |
| Settled cost | Billing reconciliation after usage posts; not a second-by-second measurement |

Poll queue/replica telemetry every second where supported, with backoff and freshness timestamps. A stale metric remains labelled stale. Submitted/running/completed ledger counts remain available independently; never relabel them as measured Baseten queue depth. Async Chain responses do not stream frames: the application event feed supplies progressive tile updates. [Queue schema](https://docs.baseten.co/api-reference/non-regional/get-async-queue-status-for-the-production-environment), [Chain deployment schema](https://docs.baseten.co/reference/management-api/deployments/gets-a-chain-deployment-by-id), [metrics](https://docs.baseten.co/observability/metrics)

### Full-scale performance target and cost definitions

Retain **1,500 episodes, 60 seconds, approximately $11** as the target, with **100 world-model replicas as the initial capacity hypothesis**. `max_replica` defaults to 1 and is configurable; a configured maximum is neither an active replica count nor reserved GPU capacity. Verify account capacity and prewarm all bottleneck Chainlets. Preserve the live 1→100 scale-up experiment as a separately timed measurement; do not hide cold starts inside a supposedly instantaneous transition. [Autoscaling](https://docs.baseten.co/deployment/autoscaling/overview), [cold starts](https://docs.baseten.co/deployment/autoscaling/cold-starts)

Sanity checks, not measurements: under the old five-chunk assumption, 1,500×5 calls on 100 single-request replicas require at most **0.8 seconds/chunk** to finish in 60 seconds even with zero other overhead. The old example of 12.4 GPU-seconds/episode implies 18,600 GPU-seconds total, at least 186 seconds on 100 GPU equivalents, or 310 GPU equivalents averaged over one minute. Its $0.0075/episode implies $11.25 total and about $2.18/GPU-hour if paired with that compute figure. Those example values were not measured; they must not appear as achieved telemetry. Recompute capacity from certified feedback cadence and the 70/70/100/100/80 horizons, not the obsolete five-chunk loop.

Maintain two cost views:

- **Marginal execution estimate:** attributed policy/world/validity/judge work, retries and transfers for this run. It may omit idle capacity and is explicitly labelled a marginal estimate.
- **Total demonstration-run cost:** all allocated resources from prewarm start through cooldown, including idle/cold capacity, all stages, retries, CPU, storage and network charges. Count each allocation interval once; do not add warm-up again if already included. Report prior development/training/calibration costs separately, not as zero.

For each view, `estimated_usd = sum(resource_rate_per_hour × allocated_resource_hours) + other_charges`, using nonoverlapping allocations and explicit shared-capacity attribution. First establish the pricing basis from the vendor/account contract: currency, billing unit, resource applicability, tax/egress/storage treatment, retrieval time and reconciliation method. Convert confirmed units to hourly USD; do not assume an API field named `price` already has those units. If units or applicability are unknown, display USD as unavailable. Compare with settled usage later. Record pricing snapshot, resource type, GPU count, allocation windows and attribution method. [Account instance prices](https://docs.baseten.co/reference/management-api/instance-types/gets-instance-type-prices), [billing](https://docs.baseten.co/organization/billing)

Gate F runs three fresh 1,500-episode rehearsals using the selected confirmed cost setting and judge. Each uses a new run ID and fresh policy/world/validity/judge execution: prewarmed weights are allowed, cached actions, generated segments/videos, judge outputs and prior terminal records are not. Request IDs and artifact timestamps provide evidence. The execution timer starts at the user's button/server run acceptance, includes submission, queueing, policy/world generation, all required judge samples and durable finalization, and ends only when all planned episodes have terminal records. Every terminal record must have a persisted generation artifact or explicit service-failure record, and a scoring attempt or explicit unevaluable reason. To pass, all 1,500 must finish generation with no terminal/unresolved service failures, every eligible video must undergo judging, scientific coverage must meet frozen tolerances, elapsed time must be ≤60 seconds and total demonstration-run cost ≤$11.25 (the operational definition of “approximately $11”). Scientific exclusions can remain nullable and are not replaced with extra lucky rollouts. Report every rehearsal and its estimated versus settled cost status, including failures; visible prewarm duration and allocated replicas count toward cost. Until billing reconciliation, qualify the cost result as an estimate.

If the target fails, keep the full study and report actual time/cost and unmet constraints. Do not silently reduce it to 400 episodes, replay cached videos as live, or count accepted requests as completed rollouts. Increase capacity or improve the validated operating point only with a recorded experiment; more replicas can improve latency while increasing cost.

Use **Training Jobs** for judge distillation; framework/model compatibility is checked in its own container. Loops' access/model catalogue is not a dependency. Report GPU-seconds per logical episode, rollouts per GPU-hour and cost per thousand, plus queue and replica telemetry. Do not label diffusion throughput as tokens/sec or time-to-first-token. Engine Builder/speculative decoding are not the FD serving path; quantization and multi-node exclusions are design choices, not blanket assertions about hardware support. [Training Jobs](https://docs.baseten.co/training/overview), [LoRA research context](https://www.baseten.co/blog/practical-lora-research/)

---

## 8. Output artifacts and record contracts

```
assets.lock.json                  # immutable asset/code/runtime/access records
protocol.json                     # frozen matrix, splits, seeds, tolerances and analysis
scenarios.jsonl                   # source-state lineage, images/state/goals and hashes
results/
  gates.json                      # A–F status, evidence, thresholds and revisions
  run_ledger.jsonl                # export of durable logical episodes and attempts
  per_episode/<run_id>.jsonl      # one final record per planned logical episode
  segments/<run_id>/              # immutable frames/actions/history/RNG checkpoints
  leaderboard.md                 # rates, reference, bounds, coverage, n and parity status
  reliability.json               # split-half, MDD, drift, cost, MMRV and uncertainty
  judge_calibration.json         # split/annotation IDs, agreement, errors and leniency
  exclusions.json                # missing/invalid reasons and sensitivity by cell
  load_test.json                 # all three full rehearsal results, not best-of-three
  economics.json                 # pricing, allocations, estimates and reconciliation
  provenance/                    # source snapshots, licenses, hashes and conversions
```

Per-episode record example (illustrative placeholders, no fabricated measurements):
```json
{
  "schema_version": 1,
  "run_id": "example-only", "episode_id": "logical-episode-id",
  "cohort": "primary", "protocol_hash": "sha256:...",
  "task": "close_drawer", "start_id": "...", "start_lineage_id": "...",
  "scenario_manifest_hash": "sha256:...", "status": "planned",
  "policy": {"name": "OpenVLA", "variant": "anchor-wrapper", "asset_manifest_id": "...", "adapter_hash": "sha256:..."},
  "world_model": {"repo": "nvidia/Cosmos3-Nano", "asset_manifest_id": "...", "backend_profile_hash": "sha256:..."},
  "judge": {"name": "Qwen/Qwen2.5-VL-7B-Instruct", "asset_manifest_id": "...", "rubric_hash": "sha256:...", "sampling_hash": "sha256:..."},
  "seeds": {"policy": 1017, "world": 2017, "judge_samples": [3017, 3018, 3019, 3020, 3021]},
  "feedback_mode": "unqualified", "parity_status": "unqualified",
  "horizon_actions": 70, "executed_actions": 0, "n_segments": 0,
  "validity": "unknown", "progress_score": null, "binary_success": null,
  "missing_reason": "not_run", "raw_judge_samples_ref": null,
  "attempt_ids": [], "platform_request_ids": [],
  "segments_manifest_ref": null, "video_ref": null,
  "timing_ref": null, "compute_gpu_seconds": null,
  "allocated_gpu_seconds": null, "estimated_usd": null, "cost_basis_ref": null
}
```

`status` is `planned | submitted | running | completed | failed | cancelled`; status tracks execution, while validity and nullable labels track measurement. A completed generated video can be scientifically unevaluable. `feedback_mode` records `unqualified`, `native_feedback`, or `forecast_state`, with approximation details in the profile; it never labels simulated state as measured physical state. Finalize every planned episode, including failures; the example above is pre-run, not a valid final result.

Every artifact reference resolves to a URI, SHA-256, media/schema type and immutable source revision where applicable. The segment manifest retains raw 7-D and compiled actions, input/output frame hashes, forecast states, nominal control/frame timestamps and their mapping, policy history and RNG state. Source-data measured control times and server wall-clock events are stored separately. Preserve raw judge outputs including rejected samples. Manifest links resolve weights, code/container/normalizer revisions, source images and goal references. Timings cover submission, queue, each stage, persistence, retry and finalization. Per-episode cost allocation must reconcile to run totals without double-counting shared idle capacity. Null means unknown, never zero.

---

## 9. Build order — full scope retained

| Step | Task | Acceptance/dependency |
|---|---|---|
| 0 | Recover missing handoff material; define protocol, manifests, access and isolated environments | Explicit unknowns; immutable source records |
| 1 | Vendor/Bridge fixtures, policy adapters, action compilers and world-adapter preflights | Gate A; shape/frame/rotation/gripper/normalization tests |
| 2 | Matched scenario panels; control-feedback and action-intervention suite | Gates B/C for all six policies and five tasks |
| 3 | Baseten Chain controller, durable ledger, checkpointed segments, one-policy end-to-end trace | Real request/result persistence and failure recovery |
| 4 | Three-policy integration/soak stage: OpenVLA, MiniVLA, Octo-Small × open/close drawer and basket × 100 starts | 900 development episodes; separate cohort, no primary-study claim; acquire independent starts |
| 5 | Human annotation, blinded calibration, sampling selection and judge freeze | Gate D; 150 clips, 50 double-labelled held-out |
| 6 | Freeze preregistration; execute six policies × five tasks × 50 matched starts | 1,500 primary records with provenance and parity labels |
| 7 | Split-half, MDD, cell/reference comparisons, MMRV and exclusion sensitivity | §6; all planned episodes accounted for |
| 8 | Resolution/steps/partition cost sweep; 1–6 chunk horizon and controlled drift studies | Fixed-horizon comparisons; independent cost confirmation |
| 9 | Judge distillation on Training Jobs and fresh held-out calibration | New revision, original results preserved; Gate E |
| 10 | Full dashboard, 1,500-rollout burst, capacity/prewarm measurements and three rehearsals | Gate F for numerical speed/cost claims |
| 11 | Free-play: arrow-key control, one fullscreen tile, fixed prompt, bounded simulated actions; 480p hero | Unscored and clearly labelled; certified world adapter |
| 12 | Reverse-validation experiment | Recover original question/script or explicitly specify it with the project owner; do not invent the missing experiment |

Dashboard scaffolding and data acquisition can proceed alongside backend work. Publication gates still apply; visual polish does not bypass them. Reverse validation remains part of delivery, with the missing question/source tracked as a dependency rather than silently dropped.

---

## 10. Do NOT build

- **A checkpoint ladder.** Keep measurement reliability and MDD as the project focus; prior work already studies checkpoint ranking.
- **World-model adaptation or policy fine-tuning for improvement.** Evaluate the selected released policies/models. Judge distillation remains in scope.
- **A SIMPLER installation or environment hub.** Use the published comparator tables with their stated limits.
- **A forced six-policy ranking, unsupported “first ever” claim, or invented performance counter.** Report indeterminate results and unmet targets honestly.
- **Silent protocol substitutions.** No stale-observation action packing, unlabelled forecast state, shorter task horizons disguised as cost savings, or replacement of missing task scenes.

---

## 11. Dashboard requirements

Three regions:

1. **12 video tiles**, optionally staggered 40–60 ms as presentation timing, extending only when persisted segment events arrive. Tile identity includes run/episode/task/policy; the tile count is a viewport choice, not the total robot or policy count. Append actual certified frame counts, not hardcoded 16-frame chunks. Label cached/replayed/qualitative video explicitly.
2. **Scoreboard** with virtual rates, published reference, uncertainty, coverage, missingness bounds and indeterminate orderings. Expose parity/backend/judge revision. Separate live provisional counts from finalized study statistics; exclusions and errors remain visible.
3. **Live telemetry strip** using §7's source mapping, updating every second where available: logical episodes completed, platform queue, active/desired replicas, compute/allocation time, estimated marginal and total cost. Show metric freshness and reconciliation status.

Keep the **1,500-rollout burst button** and **cost slider**. Select only a held-out-confirmed configuration for qualified claims; the slider reads precomputed sweep results instantly without regenerating on drag. Display fixed task horizon and coverage beside each cost point so shorter tasks cannot masquerade as savings. The backend deduplicates repeated button submissions by run identity.

Keep **free-play**, arrow keys driving one fullscreen simulated arm, with release-to-stop, clamped actions, fixed instructions, visible latency, and no scoreboard contribution. Keep the **480p hero rollout** for presentation, clearly distinguished from the 256p primary scoring protocol. If a 480p setting enters scored comparisons, qualify it as its own operating point.

The 180-second sequence remains: six-clip real/generated audience test, locked comparison to the published OpenVLA drawer cell, and full-matrix burst. Do not claim audiences were fooled without recorded results or that PLUMB beat the comparator before its own estimate exists.

## 12. Evidence and open dependencies

### Execution evidence — 2026-09-19

These are diagnostic observations, not replacements for the full study or its
qualification gates. Raw reports and small media copies live under
`data/cluster-evidence`; model weights remain on the clusters.

- The SQLite execution ledger, recovery/cancellation, measurement guards,
  calibration workflow, policy/world contracts, API, and 12-tile dashboard are
  implemented and tested. A full 1,500-row **synthetic fixture** completed; it is
  not a learned-model experiment and its timing is not a burst result.
- Cosmos Nano on H100: 16 actions produced 17 frames; an actual 480-tier inference
  took 4.52 seconds, excluding 167.68 seconds of model loading (`937372`).
  Native one-action inference returned only the condition frame. Same-seed exact
  repeats were identical, while permuting future actions changed the first future
  frame (`937423`). The pinned Cosmos/OpenVLA one-step profile fails Gate B; do not
  patch over it with hidden future actions or padding.
- OpenVLA emitted a real 7-D Bridge action in 0.836 seconds (`937500`). The original
  IRASim Frame-Ada model, still instantiated at 16 frames/extras=3/mask=1, accepted
  one native action and produced two frames in 2.55 seconds (`937666`). This is an
  experimental short-horizon path, not the released 15-action evaluation protocol.
- A real 16-tick OpenVLA→IRASim loop completed (`937704`): each generated image fed
  the next policy call, with 17 persisted frames and 142.21 seconds total including
  loading/artifacts. The re-encoded-image feedback path developed severe visible
  gripper drift. It passes engineering trace checks only, not physical fidelity.
  The original schema-v1 bundle lacks its exact task/release identity and labels
  pixel hashes ambiguously; preserve it unchanged. Subsequent schema-v2 runs
  separately bind task, source release, decoded pixels, and PNG file bytes.
- Actual five-sample Qwen2.5-VL diagnostics completed on Queen's A100. The closed-loop
  clip returned unknown/insufficient quorum; no task score is asserted. Human
  calibration, development/held-out separation, and Gate D remain unsatisfied.
- IRASim action controls (`937739`) completed all 18 calls over two seeds. Exact
  repeats had zero raw-pixel MAE; directed ±XYZ changes differed from the original
  action by 1.197–3.960/255 MAE in the next frame. Held-gripper stationary commands
  also changed output. This supports reproducibility and action sensitivity on one
  scene, not correct motion direction, scene parity, or physical fidelity.
- A paired state-representation replay (`937751`) held the 16 saved actions and
  diffusion RNG stream fixed across image re-encoding and source-backed latent
  carry. First-frame pixels/latents matched exactly; same-seed repeats reproduced
  all 16 outputs in both branches. Both developed severe visible gripper drift.
  The state-transport choice changes the video (final paired MAE10.705/255), but
  latent carry alone did not resolve the observed problem. Total runtime124.80s
  includes both branches, their repeat, loading and artifacts—not one rollout.
- The released 15-action/16-frame open-loop reference (`937775`) completed in
  34.47s including loading/artifacts. Its inspected final frame is more structurally
  coherent than the short-horizon rollouts, but that is qualitative only. The
  first prediction receives 14 future action rows; this cannot certify or replace
  native OpenVLA feedback. Different horizon/noise shapes prevent a causal paired
  attribution of the visual difference.
- A causal growing-history replay (`937822`) generated one new frame from up
  to 15 committed past latents and already-executed saved actions, with no
  future-action rows. The 16-tick run and repeat took 187.265 seconds including
  loading/artifacts. All pixel and latent hashes repeated exactly, but the
  inspected final frame still had substantial blur and gripper/object
  distortion. The inference conditioning differs from training mask=1, and
  this replay did not query a policy or resolve physical fidelity.
- Octo-Small v1.0 now runs in an isolated cluster runtime. The first attempt
  exposed an API mismatch: pinned Octo v0.1 returns normalized actions, whereas
  AutoEval calls a newer unnormalization API. The corrected, distinctly named
  native-v0.1 diagnostic (`937277.1`) produced three finite four-action
  proposals with exact reset/repeat. Eight workers across two H100 nodes
  (`937277.2`) completed24 calls and matched fixed-input outputs exactly.
  This is runtime/repeatability evidence, not AutoEval replication, generated
  feedback, task evaluation, or a burst/cost qualification.
- Immutable cluster downloads pin Cosmos `e59a53c25979a090fa8706c9acc0c254a6e89b92`,
  OpenVLA `47a0ec7fc4ec123775a391911046cf33cf9ed83f`, Qwen
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, and IRASim source
  `c72b6dade6fcd65971e0aa8ab49ea39b15108c90`. Original IRASim EMA weights were safely
  converted with restricted loading to strict-verified safetensors; SDXL VAE
  licensing remains distinct from IRASim's Apache-2.0 source.

The runtime gate view is `data/gates.json`, always unqualified. Cosmos and IRASim
gate evidence is backend-specific. An IRASim smoke pass cannot clear Cosmos's
causal-feedback failure or establish matched task panels, calibration, or cost.

### Source record

The document review used four adversarial Terra xhigh reviews and three independent Terra xhigh verification reviews. It checked source claims and arithmetic, not GPU runtime behavior. URLs with commit IDs below are immutable anchors; model cards and service docs without revisions must be snapshotted when creating the asset/protocol lock.

| Topic | Evidence and disposition |
|---|---|
| Human/SIMPLER cells and horizons | [AutoEval](https://arxiv.org/html/2503.24278v2): preserve tables; correct Octo prose and leave-one-task-out claim. |
| Policy configuration | [AutoEval code](https://github.com/zhouzypaul/auto_eval/tree/3ea3ff44c6950433cfbcb4294a3deaa616533745), [OpenVLA](https://github.com/openvla/openvla/tree/c8f03f48af692657d3060c19588038c7220e9af9), [OpenPiZero](https://github.com/allenzren/open-pi-zero/tree/c3df7fb062175c16f69d7ca4ce042958ea238fb7): certify each native wrapper; identical checkpoint names do not establish identical control protocols. |
| Cosmos action conventions | [Framework](https://github.com/NVIDIA/cosmos-framework/tree/c23e51f2f157ae3e51cfcd86ebfb5464850894f2), [Nano card snapshot](https://huggingface.co/nvidia/Cosmos3-Nano/blob/e59a53c25979a090fa8706c9acc0c254a6e89b92/README.md): stateful transform; backend-dependent fields. Short-horizon/causal behavior is not established by permissive argument validation. |
| IRASim compatibility | [Pinned source](https://github.com/bytedance/IRASim/tree/c72b6dade6fcd65971e0aa8ab49ea39b15108c90): separate 7-D, 15-action, 256×320 adapter; not drop-in ready. |
| Judge identity | [V-JEPA](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256), [Qwen](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct): encoder versus rubric VLM; no robot-judge accuracy inferred from either card. |
| Baseten | Official invocation, queue, deployment, pricing and training links in §7 establish API responsibilities, not PLUMB throughput. |
| Prior evaluation work | [WorldEval](https://arxiv.org/abs/2505.19017), [WorldGym](https://arxiv.org/abs/2506.00613), [dWorldEval](https://arxiv.org/abs/2604.22152), [Runway](https://runway.com/research/accelerating-robot-policy-evaluation): substantial existing world-model policy evaluation. A dated metric-by-metric literature matrix is required before claiming priority for the proposed measurement combination. |

Rejected reviewer overcorrections: the hardcoded score tables were already correct; current model cards do not support changing Edge to 2B or Super to 32B; the reviewed Bridge source does not establish the asserted universal multiple-of-four restriction; SGLang FD support cannot be declared universally present or absent from conflicting runtime descriptions. Resolve compatibility using one pinned tested tuple.

### Missing inputs — retain features, do not invent evidence

Only this file and `PASS-OFF.md` were supplied. `BUILD-SPEC.md`, `SCRIPT.md`, `htn2026-field-guide.md`, and `reverse_validation.py` are absent. Recover them from their original source; do not describe the reverse-validation script as runnable or invent which judge/question it answers. The 180-second sequence is specified here, but a word-for-word original script is not available.

Other unresolved inputs: five-task start/goal/reference panels; fidelity-qualified world-model feedback; forecast-state validation; remaining legacy policy artifact licensing and safe conversion; frozen judge calibration; capacity/pricing; and qualified performance measurements. The implementation team records owners and evidence status in the run protocol. These are dependencies of the full build, not scope cuts.

### Acceptance checks

- Preserve both reference tables exactly; assert 1,500 primary cells/trials, Octo 2/250, MiniVLA 127/250, OpenPiZero 126/250 and the four-of-five reversal result.
- Test state transforms with nonzero orientation, action/state gripper polarity, normalization once-only, conditioning-frame removal, prompt propagation, timestamps, native feedback and exact terminal horizons; test every backend independently.
- Test action interventions and future-suffix perturbations, legitimate stationary scenes, invalid/unknown video, no action-text leakage, malformed judge output, disagreement and retries.
- Test `V=0`, all success/failure, unequal missingness, ties, constant correlation vectors, clustered starts, MMRV argument order, split leakage, and family-adjusted uncertainty. Expected partial-bound widths must match missingness.
- Test duplicate submission/callback, lost callback, resumed segment, timeout/cancellation and stale telemetry; logical completion counts and statistical denominators must not grow with retries.
- Test cost reconciliation across prewarm/execution/cooldown and all stages, full-matrix burst completion, live-versus-replay labelling and fixed-horizon cost-slider comparisons.
- Re-review both documents for contradictory counts, unsupported achieved/novelty claims, missing-reference status and dropped features. All scientific/runtime tests above remain implementation requirements, not tests performed during this document revision.

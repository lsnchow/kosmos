# Experimental judge fine-tune v1 — executed, not qualified

The user approved this development-only pilot. It is **not** a calibrated
judge, the formal distillation search, a Baseten Training Job, or a primary
study result. All original media, teacher outputs, failed attempts and both
trained checkpoints are retained. Model/adapter weights remain cluster-only.

## Actual execution

| Stage | Trillium job | Result |
| --- | --- | --- |
| Teacher startup | 939398 | Failed before inference; manifest-schema mismatch, fixed |
| Teacher collection | 939423 | Completed;16 clips,80 sample slots,141 actual attempts |
| LoRA optimization | 939451 | Completed;2 epochs,8 actual optimizer steps |
| Saved-adapter reload | 939458 | Completed;paired base/adapter generation on3 development clips |

Input selection was frozen before labelling:16 distinct real Bridge
close-drawer videos at revision`0e9d76d07e9df3ea3eba257b2520d4913833fad2`,
12 training /4 development-validation assignments. Original video and Parquet
downloads total2,666,878 bytes. Each example uses16 unique lossless sampled
frames, source Parquet nominal timestamps, and an initial **scene** reference,
not an invented goal. All16 source lineages remain excluded from future formal
training, calibration and evaluation, including clips whose labels abstained.

Only7 labels survived v1's unique-modal-tuple rule:4 train /3 validation;
9 tied cases were excluded without reassignment. The rule was too permissive:
every accepted clip had only1 valid sample out of5. All7 retained targets are
`[artifact, visible, 5, met]`, all from sample2. Across the80 sample slots,
30 final samples parsed and50 did not. No accepted target had3/5 support.

The runtime's decisive-label rule requires integrity=`intact`; these `artifact`
targets cannot establish task success. They remain uncalibrated structured
teacher samples, not human labels or quorum outcomes.

## Optimizer and saved adapter

Qwen2.5-VL-7B revision`cc594898137f460bfe9f0759e9844b3ce807cfb5`, LoRA rank64,
alpha32, q/v projection targets, learning rate1e-4, batch1, two manual AdamW
epochs, seed20260919. Initial development loss was0.7981337; epoch1 loss0.6012993;
epoch2 loss0.3868534. Epoch2 was selected using development loss only.
Peak CUDA allocation20,569,163,776B. The10.3583s loop timing excludes manifest
verification/model setup; Slurm job elapsed46s. No dollar estimate is asserted.

Selected adapter directory:
`/scratch/lchow432/plumb/experiments/judge-lora-pilot-v1-trained/adapters/epoch-02`

Selected adapter tree SHA-256:
`7a0b7796a0ac8b78fbb8ba8f4c4db01fe8d35c2edc3cb7a0fbeec9f4fb20523e`.
Teacher/training source release:
`cacf20c6120b4511c6bb5a2fe76c249dea2842b61d538bcfec26a87b59f1832d`.
Reload-comparison release:
`c7d025d991deb911f8e8d31fe85df9fa31fed2230a46edb6e889b8e925aba9c8`.

## Reload result and why v1 must not be scaled

The saved adapter was loaded afresh through PEFT and paired with its disabled-
adapter base on the same3 assigned development-validation clips. All6 outputs
were schema-valid. The adapter repeated`[artifact,visible,5,met]` on3/3;
the base reported`intact` on3/3. Thus full semantic agreement was0/3, while
both reported progress5/completion met. Lower loss demonstrates teacher fit,
not better judgement; the pilot copied a homogeneous, non-quorum label set.

Do not deploy or scale v1. The next experiment is a separately named v2
**prompt/selection diagnostic** with explicit visual-integrity/intended-contact
definitions and at least3/5 identical valid semantic votes. No old label will
be repaired, overwritten, or promoted. Any formal use still needs fresh,
stratified, independently human-labelled calibration data.

Small local evidence is in `data/judge-pilot-v1-evidence/`,
`data/live-integrated/cluster-evidence/judge-lora-pilot-939451/`, and
`data/live-integrated/cluster-evidence/judge-lora-compare-939458/`.
The raw teacher summary incorrectly says`completed_clip_count:0` because it
counted the resume cache before collection. All16 completed raw reports and
their hashes were separately checked; raw summary preserved, code fixed.

# Policy-loader dependency record

This record names source and artifact dependencies for the local MiniVLA and
SuSIE loaders. A configured loader is not a qualification result: every arm
still requires its own fixtures and Gate A/B evidence.

## MiniVLA

- Code: [Stanford-ILIAD/openvla-mini `0822b36227b5a771be4eb2680e34c559734c8fdc`](https://github.com/Stanford-ILIAD/openvla-mini/tree/0822b36227b5a771be4eb2680e34c559734c8fdc).
  Its [local VLA loader](https://github.com/Stanford-ILIAD/openvla-mini/blob/0822b36227b5a771be4eb2680e34c559734c8fdc/prismatic/models/load.py#L130-L257), [action API](https://github.com/Stanford-ILIAD/openvla-mini/blob/0822b36227b5a771be4eb2680e34c559734c8fdc/prismatic/models/vlas/openvla.py#L37-L121), and [VQ decoder](https://github.com/Stanford-ILIAD/openvla-mini/blob/0822b36227b5a771be4eb2680e34c559734c8fdc/prismatic/vla/action_tokenizer.py#L110-L180) are the reviewed API boundary.
- Main model: `Stanford-ILIAD/minivla-vq-bridge-prismatic` revision
  `931a637fbfc783220df9c47eb613bf6c9c6e1c4a`; required checkpoint is
  `checkpoints/step-362500-epoch-21-loss=0.2259.pt`, SHA-256
  `2b1828f4fb96b0b7a4f3d191fde4ee96938b293c70f8616fb44dd85f5c85cadc`.
  The required config and dataset-statistics SHA-256 values are
  `a241c94667d023877ee11f872bac65c3107edc2b0509fc88a4fa0aa7f3bcebce`
  and `49742ae0009501e1ab283641ab1794a218cee41401cc0c978ed32cc5551adb4f`.
- Required VQ artifact: `Stanford-ILIAD/pretrain_vq` revision
  `30ef2227f97dde1abb6d522ea80af85384235008`,
  `pretrain_modvq+mx-bridge_dataset+fach-7+ng-7+nemb-256+nlatent-512/checkpoints/model.pt`,
  SHA-256 `4c3ac8c89f9f092b8b2055b20ed6113a45b7359c5bcd0244f4e725ab5b2a7ce2`.
  **Blocker:** that immutable snapshot has `cardData: null` and declares no
  license. Do not acquire, copy, convert, redistribute, or use it until the
  publisher supplies terms and the asset lock records them.
- The source VQ tokenizer depends on [VQ-Bet
  `09d4851288ca5deaaa1ab367a208e520f8ee9a84`](https://github.com/jayLEE0301/vq_bet_official/tree/09d4851288ca5deaaa1ab367a208e520f8ee9a84).
  PLUMB's loader bypasses its unsafe `torch.load` convenience path and requires
  `torch.load(..., weights_only=True)`, exact digests, a tensor-only state tree,
  and strict state-dict loading.
- Runtime tuple: Python 3.10, Torch 2.2.0, Torchvision 0.17.0,
  Transformers 4.40.1, Tokenizers 0.19.1, timm 0.9.10, Flash Attention 2.5.5.
  All required assets must already be local; hub access is forced offline.
- A real profile also names clean local `openvla-mini` and `vq_bet_official`
  checkout paths. PLUMB verifies their exact Git HEADs, empty working trees,
  reviewed source files, and that imported `prismatic`/`vqvae` classes resolve
  inside those checkouts. Merely repeating a commit string in a profile is not
  source proof.
- **Additional construction blocker:** the reviewed MiniVLA source creates its
  DINO and SigLIP featurizers via `timm.create_model(..., pretrained=True)` and
  its Qwen2.5 backbone through a source helper. `HF_HUB_OFFLINE=1` does not
  prove those helpers cannot reach a timm/torch-hub URL. No source-supported,
  local DINO/SigLIP/Qwen cache bindings with verified checksums are presently
  supplied, so PLUMB blocks real MiniVLA construction even after VQ licensing
  is resolved. It does not try the helpers or add a generic network patch.
- Use `unnorm_key="bridge_dataset"`, not OpenVLA's `bridge_orig`.
  The released VQ config has `input_dim_h=8` (seven future actions), but its
  source decoder returns only `ret_action[:, 0]`; it therefore exposes one
  7-D action. It cannot be advertised as an exposed seven-action proposal until
  that mismatch is resolved with source-backed fixture evidence.

## SuSIE and SuSIE_LL

- AutoEval replication source: [policy wrapper
  `3ea3ff44c6950433cfbcb4294a3deaa616533745`](https://github.com/zhouzypaul/auto_eval/blob/3ea3ff44c6950433cfbcb4294a3deaa616533745/auto_eval/robot/policy.py#L454-L652)
  and [configuration](https://github.com/zhouzypaul/auto_eval/blob/3ea3ff44c6950433cfbcb4294a3deaa616533745/scripts/configs/eval_config.py#L56-L111).
  Its named replication arm is `susie-autoeval-gc-bc-replication` and uses
  `gc_bc`, a 20-policy-step subgoal cadence, 256×256 RGB images, deterministic
  low-level argmax, normal action unnormalization, and a binary gripper.
- Upstream SuSIE source: [README
  `8177f63332a3905202c122310a92c51b8ff14280`](https://github.com/kvablack/susie/blob/8177f63332a3905202c122310a92c51b8ff14280/README.md#L1-L28)
  specifies `gc_ddpm_bc`, horizon 4, and `delta_goals` for the low-level
  policy. The separate sensitivity ID is
  `susie-upstream-gc-ddpm-bc-sensitivity`; it must never replace or pool with
  the AutoEval replication arm. Its agent/config source is [BridgeData V2
  `bc60a35b701a12021c8c95e9d8601274d3acd928`](https://github.com/rail-berkeley/bridge_data_v2/tree/bc60a35b701a12021c8c95e9d8601274d3acd928).
- The low-level-only IDs are `susie-ll-autoeval-gc-bc-replication` and
  `susie-ll-upstream-gc-ddpm-bc-sensitivity`. SuSIE_LL requires a
  scenario-provenance goal image; it does not turn task text into a fabricated
  goal image.
- Released high-level UNet: `kvablack/susie` revision
  `83d3c21fda79550fc5431f8a42c07835c9ec3f95`,
  `unet/diffusion_flax_model.msgpack`, SHA-256
  `371d791b6e605a4a6d927fe704172e5023e83661affb1693ff421c2095f437a2`;
  required `unet/config.json` SHA-256 is
  `5dd4c5adb5bb76946e821c1a4ab362c0686861311ddb152926cf66052d0b2cf4`.
  Released AutoEval low-level component: `patreya/gcbc-bridge` revision
  `1a4c15dd9ad780a257e9494f0fac79cbe8e64793`,
  `checkpoint/checkpoint`, SHA-256
  `80b354db7a05d514d6df5b5a4395469902b0ff362d383edeeb4abb8c1c9d9e33`.
  That component is `gc_bc`; it is not evidence or weights for the corrected
  `gc_ddpm_bc` arm.
- The high-level path also needs a separately pinned, local,
  license-recorded `lodestones/stable-diffusion-v1-5-flax` snapshot. AutoEval
  did not pin its revision, so PLUMB refuses high-level loading until the
  revision, asset digest, and license evidence are in the profile/asset lock.
  The profile supplies a local Stable Diffusion asset-manifest file and PLUMB
  hashes that file against the recorded digest *and* verifies SHA-256 records
  for every regular file in the source-loaded `vae/`, `text_encoder/`, and
  `tokenizer/` subtrees. Paths outside those trees, symlinks, unrecorded files,
  or revision/model-ID mismatches block loading. It also requires clean local SuSIE, BridgeData V2, and AutoEval
  checkouts with exact reviewed HEADs, and verifies that imported sampler and
  JAXRL agent code resolve inside the corresponding checkout.
- Runtime tuple from the source requirements: Python 3.10, JAX 0.4.11, Flax
  0.7.0, Diffusers 0.18.2, Transformers 4.33.1, plus the source-pinned
  BridgeData/JAXRL stack. The high-level source initializes its JAX RNG from
  wall-clock time; this limitation is reported rather than relabelled as a
  deterministic seeded rollout.
  PLUMB passes `wandb_run_name=None` and rejects high-level roots containing
  `checkpoint`, preventing `create_sample_fn` from taking its W&B/Orbax branch.

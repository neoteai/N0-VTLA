<div align="center">

<h1>𝒩₀-VTLA</h1>

<p><b>Scaling Vision-Tactile-Language-Action Model with Latent Tactile Tokens</b></p>

<p>
  <a href="https://research.neoteai.com/">NeoteAI Team · <strong>新智具身</strong></a>
  &amp; <a href="https://teai.fudan.edu.cn/">Fudan TEAI Team</a>
</p>

<p>
  <b>English</b> &nbsp;·&nbsp; <a href="README_CN.md">中文</a>
</p>

<p>
  <a href="https://research.neoteai.com/n0-vtla/"><img src="https://img.shields.io/badge/Project-Page-1f6feb.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2607.23782"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper"></a>
  <a href="https://huggingface.co/NeoteAI/n0-vtla-base"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-n0--vtla--base-ffce3a.svg" alt="Weights"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--SA%204.0-green.svg" alt="License: CC BY-SA 4.0"></a>
  <a href="mailto:hengzzzhou@gmail.com"><img src="https://img.shields.io/badge/Contact-hengzzzhou%40gmail.com-0a66c2.svg" alt="Contact: hengzzzhou@gmail.com"></a>
  <img src="https://img.shields.io/badge/Python-3.11-3776ab.svg" alt="Python 3.11">
</p>

<h2>🔥 𝒩₀-VTLA Has Been Released! 🔥</h2>

<p><strong>Pretrained checkpoints, the post-training toolkit, and inference servers are now available.</strong></p>

<img src="docs/media/vtla-teaser.webp" width="860" alt="N0-VTLA: large-scale visuo-tactile pretraining with latent tactile tokens">

</div>

$\mathcal{N}_0$-VTLA is a vision–tactile–language–action policy that gives a pretrained
vision–language–action (VLA) model a sense of touch. Rather than feeding tactile images in as
extra camera streams, it **predicts latent tactile tokens**, a compact code for the contact
change expected over the next action chunk, and conditions a flow-matching action expert on them
directly. Force, slip, and the difference between alignment and jamming are visually subtle but
directly observable through touch; the latent tactile pathway makes that signal available to the
policy.

This repository releases the model, the post-training toolkit, and the inference servers, so you
can load our pretrained weights and adapt $\mathcal{N}_0$-VTLA to your own robot and tasks. It
does not ship the large-scale pretraining pipeline.

## Capabilities

- Post-train the pretrained checkpoint on your own visuo-tactile demonstrations with `train.sh`. See [POST_TRAINING.md](docs/POST_TRAINING.md).
- Convert your episodes to the canonical 32-dim layout and compute normalization statistics (`scripts/convert_canonical_data.py`, `scripts/compute_canonical_norm.py`).
- Serve a policy over websocket for a real robot, or over ZMQ for the simulator (`scripts/serve_policy.py`, `scripts/serve_zmq.py`). See [DEPLOY.md](docs/DEPLOY.md).
- Check that the tactile pathway is doing something, and that the model stays byte-equivalent to the base policy with tactile off (`scripts/probe_z_tactile_dependence.py`, `scripts/gate_c_check.py`).

## Model summary

| Component | Choice |
|---|---|
| Backbone | PaliGemma (gemma_2b) prefix + Gemma (300m) action expert |
| Objective | Flow matching over a 50-step action chunk |
| Tactile encoder | Frozen DINOv2 (`facebook/dinov2-base`) over baseline-difference images |
| Tactile pathway | Cross-attention predictor → 5 latent tokens, injected into the action expert |
| Action space | Canonical 32-dim container (dual-arm EEF position + rot6d + gripper) |
| Precision | bf16 parameters, eager attention to match the pretraining path |

<p align="center">
  <img src="docs/media/vtla-model-simple.webp" width="860" alt="Latent tactile model overview">
  <br><em>Each tactile frame is differenced against a zero-contact baseline, encoded by a frozen
  DINOv2, and distilled into a few latent tokens summarizing the contact change expected over the
  next action chunk.</em>
</p>

The tactile path is **off by default and byte-equivalent** to the base policy when disabled, so an
$\mathcal{N}_0$-VTLA checkpoint loads and runs like the standard base policy until the tactile
branch is switched on. `scripts/gate_c_check.py` asserts that equivalence on state-dict keys,
forward loss, and sampled actions.

## Repository structure

```
N0-VTLA/
├── n0vtla/                 # the model package
│   ├── models_pytorch/     # N0VTLAPolicy, tactile encoder/predictor, transformers_replace
│   ├── policies/           # input/output transforms per embodiment, canonical schema
│   ├── training/           # configs, data loader, optimizer
│   └── transforms.py       # normalization, delta actions, tokenization
├── n0vtla_client/          # robot-side client library
├── scripts/                # train, serve, convert, verify
├── docs/                   # installation, post-training, deployment
└── train.sh                # post-training launcher
```

## Installation

Python 3.11 with a CUDA-capable GPU. The `transformers_replace` patch and the DINOv2 cache are
both required; [INSTALL.md](docs/INSTALL.md) has the full procedure.

```bash
conda create -n vtla python=3.11 -y && conda activate vtla
pip install -r requirements.txt && pip install -e . --no-deps
cp -r n0vtla/models_pytorch/transformers_replace/* \
  "$CONDA_PREFIX/lib/python3.11/site-packages/transformers/"
```

## 📦 Model Download

| Checkpoint | What it is | Config |
|---|---|---|
| [🤗 n0-vtla-base](https://huggingface.co/NeoteAI/n0-vtla-base) | pretrained base, the starting point for post-training | `vtla_tactile_posttrain` |
| [🤗 n0_VTLA_insert_hole](https://huggingface.co/NeoteAI/n0_VTLA_insert_hole) | UniVTAC single-arm policy, joint actions | `sim_single_arm_tactile` |
| [🤗 n0_VTLA_dual_bowl_place_stack](https://huggingface.co/NeoteAI/n0_VTLA_dual_bowl_place_stack) | NeoSim dual-arm policy, joint actions | `sim_dual_arm_tactile` |

```bash
hf download NeoteAI/n0-vtla-base --local-dir checkpoints/n0-vtla-base
export VTLA_PRETRAINED_CHECKPOINT="$PWD/checkpoints/n0-vtla-base"
```

`VTLA_PRETRAINED_CHECKPOINT` must name the directory that **directly** contains
`model.safetensors`.

The base checkpoint is a **pretrained base for post-training, not a deployable policy.** It
carries the tactile encoder, latent tactile predictor, and projection parameters that a task
policy needs, but it has not been fine-tuned on any task.

## Quick start: post-train on your own data

```bash
export VTLA_DATASET_PATH=/path/to/datasets/canonical_tactile_task
export VTLA_ASSET_ID=canonical_tactile_task
export EXP_NAME=my_experiment

CHECK_ONLY=1 bash train.sh     # preflight: GPUs, checkpoint, dataset, norm stats, DINOv2 cache
bash train.sh --overwrite
```

[POST_TRAINING.md](docs/POST_TRAINING.md) covers the four stages (convert episodes, compute
normalization statistics, configure, train), along with the canonical 32-dim layout and a
troubleshooting section.

## Serving

```bash
# real robot, websocket
python scripts/serve_policy.py \
  --policy.config=vtla_tactile_posttrain \
  --policy.dir=checkpoints/vtla_tactile_posttrain/<experiment>/<step>

# UniVTAC simulator, ZMQ
VTLA_ASSET_ID=n0_insert_hole_norm python scripts/serve_zmq.py \
  --config sim_single_arm_tactile \
  --ckpt checkpoints/n0_VTLA_insert_hole \
  --addr "tcp://*:5557" --default-prompt "insert hole"
```

Execute the **full** action chunk before requesting the next prediction. A shorter execution
stride drops the tail of every chunk, and the gripper-close commands live in that tail. See
[DEPLOY.md](docs/DEPLOY.md).

## Highlights

- **Predictive touch, not pixels.** Tactile evidence enters as a small set of *predicted* latent
  tokens for upcoming contact change, rather than as raw images decoded through the vision tower.
- **Baseline-difference input.** Each tactile frame is subtracted from the episode's zero-contact
  baseline, so the encoder sees contact *change* rather than the raw gel image.
- **Zero-initialized gate.** The latent tokens enter the action expert through a gate that starts
  at zero, so training begins from the exact base-policy behavior and opens the tactile channel
  gradually.
- **One action space for many robots.** A fixed 32-dim container lets single-arm, dual-arm, and
  handheld-gripper embodiments share a single action objective.

Large-scale pretraining on NeoData, the three-stage training recipe, the deployment-time (RL)
improvement pipeline, and the experimental results are described in the
[paper](https://arxiv.org/abs/2607.23782).

## Contact

For questions about $\mathcal{N}_0$-VTLA, you can contact [hengzzzhou@gmail.com](mailto:hengzzzhou@gmail.com).

## Citation

```bibtex
@misc{n0vtla2026,
      title={$N_0$-VTLA: Scaling Vision-Tactile-Language-Action Model with Latent Tactile Tokens}, 
      author={NeoteAI Team and Fudan TEAI Team},
      year={2026},
      eprint={2607.23782},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.23782}, 
}
```

## Acknowledgments

$\mathcal{N}_0$-VTLA builds on the [OpenPI](https://github.com/Physical-Intelligence/openpi)
framework, [DINOv2](https://github.com/facebookresearch/dinov2) for tactile encoding, and
[LeRobot](https://github.com/huggingface/lerobot) for data handling.

## License

**Original material** in this repository is released under
[CC BY-SA 4.0](LICENSE). Third-party components retain their original licenses; the
Apache-2.0-licensed components are identified in [NOTICE](NOTICE), with the full licence text in
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

**Model weights are not covered by that licence.** The released $\mathcal{N}_0$-VTLA checkpoints
derive from Google's PaliGemma/Gemma parameters, so they are made available under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). This is
inherited from the base model, not a restriction we add on top.

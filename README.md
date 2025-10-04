# Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study

## Introduction

Large Language Models (LLMs) rely on safety alignment to produce socially acceptable responses. However, this behavior is known to be brittle: further fine-tuning, even on benign or lightly contaminated data, can degrade safety and reintroduce harmful behaviors. A growing body of work suggests that alignment may correspond to identifiable directions in weight space, forming subspaces that could, in principle, be isolated or preserved to defend against misalignment. In this work, we conduct a comprehensive empirical study of this perspective. We examine whether safety-relevant behavior is concentrated in specific linear subspaces, whether it can be separated from general-purpose learning, and whether harmfulness arises from distinguishable patterns in activations. Across both weight and activation spaces, our findings are consistent: subspaces that amplify safe behaviors also amplify useful ones, and prompts with different safety implications activate overlapping representations. Rather than residing in distinct directions, we show that safety is highly entangled with the general learning components of the model. This suggests that subspace-based defenses face fundamental limitations and underscores the need for alternative strategies to preserve safety under continued training. We corroborate these findings with multiple experiments on five open-source LLMs from the Llama and Qwen families.


## Environment Setup

We use Conda to manage the Python environment. Set up your environment with:

```bash
conda create -n safety-spaces python=3.10
conda activate safety-spaces
pip install -r requirements.txt
```

## Experiments

### 1. Fine-Tuning

Generate baseline models with various training objectives:

```bash
bash scripts/finetune.sh
```

This creates fully useful, fully harmful, and contaminated model variants for our experiments.

### 2. Projection Analysis

Compute model projections across different SVD fraction values:

```bash
python scripts/projection.py
```

This generates a comprehensive CSV with utility and harmfulness metrics for each projection configuration.

### 3. Subspace Updating

Run:

```bash
bash exp-3-update_spaces/update_spaces.sh
```

### 4. Activation Space Analysis

Run:

```bash
python exp-4-activation_spaces/activation_space.py
```


## Citation

If you use our work, please cite us:

```
@article{ponkshe2025safety,
  title={Safety Subspaces are Not Distinct: A Fine-Tuning Case Study},
  author={Ponkshe, Kaustubh and Shah, Shaan and Singhal, Raghav and Vepakomma, Praneeth},
  journal={arXiv preprint arXiv:2505.14185},
  year={2025}
}
```

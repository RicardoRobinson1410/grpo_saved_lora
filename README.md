# Generative AI Fundamentals: Teaching an LLM to Reason (GRPO + LoRA)

This project fine-tunes **Qwen2.5-3B-Instruct** to perform procedural, step-by-step reasoning on letter-counting tasks using **Group Relative Policy Optimization (GRPO)** and **Low-Rank Adaptation (LoRA)** via **Unsloth** and **vLLM**.

---

## Project Overview

Standard LLMs often struggle with strict algorithmic tasks such as counting occurrences of a letter in a word. To overcome this "reasoning gap", the model is trained with reinforcement learning to:
1. Spell the target word letter by letter on sequential numbered lines.
2. Maintain an accurate cumulative running count at each step.
3. Enforce XML formatting (`<reasoning>...</reasoning><answer>...</answer>`).
4. Output the exact final integer count.

---

## Methodology & Architecture

- **Base Model:** `Qwen/Qwen2.5-3B-Instruct` (loaded in 4-bit precision with FP16 compute via Unsloth).
- **LoRA Hyperparameters:**
  - `lora_rank`: `64`
  - `lora_alpha`: `64` (scaling factor $\alpha/r = 1$)
  - `target_modules`: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` covering both attention and MLP layers.
  - Trainable parameters: 119.7M (~3.74% of the 3.2B base parameters).
- **Reinforcement Learning (GRPO):**
  - Group size (`num_generations`): 4 completions per prompt.
  - `learning_rate`: `1e-5` with cosine schedule and warmup.
  - `beta`: `0.0001` (KL divergence penalty to prevent policy drift).
  - Checkpointing: Saved every 20 steps (`save_steps=20`, `save_total_limit=2`).
- **Reward Functions:**
  1. `numbering_reward_func`: Rewards strictly sequential numbering (`1.`, `2.`, ...); penalizes out-of-order or missing steps.
  2. `spelling_reward_func`: Rewards exact character match with the target word; penalizes character omissions, additions, or reorderings.
  3. `counting_reward_func`: Verifies the running count matches ground truth at every character step.
  4. `format_reward_func`: Enforces `<reasoning>` and `<answer>` XML tags and integer answers.
  5. `correct_answer_reward_func`: Provides strong reward (+2.0) for the correct final answer, and negative reward (-1.0) for an incorrect answer.

---

## Training Results

- **Run Duration:** 100 steps completed on an NVIDIA T4 GPU (~29.6 minutes).
- **Correctness Reward Trend:**
  - Early mean (first 20 observations): **1.3531**
  - Late mean (last 20 observations): **1.7188**
  - Mean improvement: **+0.3656**, demonstrating clear convergence toward correct procedural reasoning.
- **Catastrophic Forgetting Probes:** Tested across 3 general knowledge questions (Manila, Paris, Mars). Both the baseline and fine-tuned models answered correctly, confirming no degradation of foundational factual knowledge.

---

## Repository Structure

```
├── README.md                                             <- Project summary and overview
├── requirements.txt                                      <- Pinned dependencies for reproducible setup
├── starter/
│   ├── README.md
│   └── gen_ai_fundamentals_project_starter.ipynb        <- Fully executed notebook with outputs and plots
├── project/starter/
│   └── gen_ai_fundamentals_project_starter.ipynb        <- Mirrored notebook at Udacity workspace path
└── grpo_saved_lora/                                     <- Exported LoRA adapter and tokenizer files
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── chat_template.jinja
    ├── tokenizer.json
    └── ...
```

---

## Getting Started

1. **Environment Setup:**
   ```bash
   pip install -r requirements.txt --no-deps
   ```
2. **Run Notebook:** Open `starter/gen_ai_fundamentals_project_starter.ipynb` in a Jupyter / Colab environment with an NVIDIA GPU (16 GB VRAM recommended).

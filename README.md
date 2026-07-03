# Intrinsic vs. Extrinsic Security Strengthening for Small Language Models

An independent research project comparing two defensive approaches against adversarial prompting on Small Language Models (SLMs) under 10B parameters.

- **Intrinsic defense**: LoRA Supervised Fine-Tuning (SFT) on Mistral-7B-Instruct-v0.2
- **Extrinsic defense**: Llama Guard 3 8B as an input/output filter

Tested across four adversarial attack techniques: Direct Injection, Indirect Injection, Persona Hijacking, and Hypothetical Framing.

---

## Key Results

| Method | Defense Success Rate (DSR) | False Refusal Rate (FRR) |
|---|---|---|
| Baseline (no defense) | 6.49% | 0.00% |
| Llama Guard 3 (extrinsic) | 66.23% | 66.67% |
| LoRA SFT (intrinsic) | **90.91%** | **17.07%** |

LoRA SFT outperforms Llama Guard 3 on both safety and usability. It also has a significant practical advantage: extrinsic defense requires running two models simultaneously (Mistral 7B + Llama Guard 3 8B), which demands approximately 14 to 16GB of VRAM and is therefore not viable for consumer hardware deployment. Intrinsic fine-tuning requires only one model at inference time.

---

## Repository Structure

```
.
├── dataset/
│   ├── direct_injection.jsonl          # Adversarial prompts - Direct Injection
│   ├── indirect_injection.jsonl        # Adversarial prompts - Indirect Injection
│   ├── persona_hijacking.jsonl         # Adversarial prompts - Persona Hijacking
│   ├── hypothetical_framing.jsonl      # Adversarial prompts - Hypothetical Framing
│   ├── sft_dataset.jsonl               # SFT training corpus
│   ├── general_question.jsonl          # General eval prompts for SFT
│   ├── lora_question.jsonl             # LoRA-specific eval prompts
│   └── OOD/
│       ├── ood_corpus_clean.jsonl      # OOD evaluation corpus
│       ├── OOD_Deepseek_hard_clean.jsonl  # OOD hard evaluation corpus
│       └── xstest_safe.jsonl           # Safe prompts for FRR measurement
├── 1_Baseline_Evaluation.py
├── 2_SFT_LoRA_Mistral_7B.py
├── 3_Extrinsic_vs_Intrinsic.py
└── README.md
```

---

## Setup

### Requirements

All notebooks are designed to run on **Google Colab with an A100 GPU**.

### HuggingFace Access

Some models used in this project are gated and require HuggingFace approval:

| Model | Gated | Required in |
|---|---|---|
| mistralai/Mistral-7B-Instruct-v0.2 | No | Notebooks 1, 2, 3 |
| Qwen/Qwen3-8B | No | Notebook 1 |
| lmsys/vicuna-7b-v1.5 | No | Notebook 1 |
| meta-llama/Meta-Llama-3.1-8B-Instruct | **Yes** | Notebook 1 |
| meta-llama/Llama-Guard-3-8B | **Yes** | Notebook 3 |

To request access, visit each model's page on [huggingface.co](https://huggingface.co) and accept the usage terms.

### Adding your HuggingFace Token to Colab

1. Open your Colab notebook
2. Click the key icon on the left sidebar (Secrets)
3. Add a new secret with the name `HF_TOKEN` and paste your token as the value
4. Enable notebook access for the secret

### Updating the GitHub path

In each notebook, update `GITHUB_RAW` to point to your repo:

```python
GITHUB_RAW = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/dataset"
```

### Google Drive path

The notebooks use Google Drive to save and load the LoRA adapter and evaluation results. Update `BASE_DIR` in each notebook to match your Drive folder:

```python
BASE_DIR = "/content/drive/MyDrive/YOUR_FOLDER"
```

---

## Notebooks

### 1. Baseline Evaluation (`1_Baseline_Evaluation.py`)

Evaluates four SLMs against the adversarial prompt corpus without any defense applied. This establishes the baseline DSR used as the reference point for all comparisons.

- HuggingFace login: **Required** (Llama 3.1 is gated)
- Dataset: loaded from GitHub
- Results: saved to Google Drive

### 2. SFT LoRA Training and Evaluation (`2_SFT_LoRA_Mistral_7B.py`)

**Part A** fine-tunes Mistral-7B-Instruct-v0.2 using LoRA SFT on the adversarial safety corpus. Training runs in full BF16 precision to maximize weight update quality.

**Part B** evaluates the fine-tuned model using 4-bit NF4 quantization with BF16 compute dtype to simulate a more realistic inference environment.

- HuggingFace login: **Optional** (Mistral is not gated)
- Dataset: loaded from GitHub
- Adapter: saved to and loaded from Google Drive

LoRA configuration:
- Rank: 16
- Alpha: 32
- Target modules: q_proj, v_proj
- Dropout: 0.05
- Epochs: 5
- Learning rate: 2e-4
- Early stopping patience: 3

### 3. Extrinsic vs. Intrinsic Evaluation (`3_Extrinsic_vs_Intrinsic.py`)

Runs the final head-to-head comparison across three conditions: Baseline, LoRA SFT (intrinsic), and Llama Guard 3 (extrinsic). Evaluates both DSR on adversarial prompts and FRR on safe prompts.

- HuggingFace login: **Required** (Llama Guard 3 is gated)
- Dataset: adversarial prompts and OOD evaluation files (from `dataset/OOD/`) loaded from GitHub
- Results: saved to Google Drive

---

## Notes on Quantization and VRAM

Training (notebook 2 Part A) uses full BF16 precision with no quantization to maximize model quality during weight updates.

Evaluation (notebooks 2 Part B and 3) uses 4-bit NF4 quantization with `bnb_4bit_compute_dtype=torch.bfloat16`. This means weights are stored in 4-bit at rest but dequantized to BF16 during computation, which is why observed VRAM usage reflects BF16 levels rather than true 4-bit levels.

Running both Mistral 7B and Llama Guard 3 8B simultaneously in notebook 3 requires approximately 14 to 16GB of VRAM, which exceeds most consumer GPU budgets (typically 8 to 12GB). This is one of the core practical findings of the research: intrinsic defense via LoRA SFT is not only more effective but also the only approach that is realistically deployable on consumer hardware.

---

## Author

**Mickhael Niklas**
Informatics Student, Pradita University
mickhaelniklas1@gmail.com
[github.com/Heatbless](https://github.com/Heatbless)

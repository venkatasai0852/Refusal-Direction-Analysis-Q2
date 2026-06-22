# Does the Refusal Direction Encode Harm or Production Intent?
### Evidence from the Mistral Model Family

> **Authors:** Venkata Sai & Urja  
> **Supervisor:** Prof. Preetam Kumar, IIT Patna  
> **Guidance:** Shambavi, WCR Lab  
> **Institution:** Indian Institute of Technology Patna

---

## Overview

This repository contains the full codebase, data pipeline, experimental results, and research report for our investigation into the **mechanistic interpretability** of the refusal direction in instruction-tuned large language models.

We address a central open question from Arditi et al. (2024):

> *Does the refusal direction encode a general concept of harmful content (H1), or does it specifically encode the act of being asked to produce harmful content (H2)?*

Using a **six-condition experimental design** across two Mistral model scales (7B and 12B), we find that **neither hypothesis is sufficient**. The refusal direction is a **compound feature** — it encodes both the semantic concept of harm and a strong contextual scaling factor tied to instructional framing.

---

## Key Findings

| Metric | Mistral-7B | Mistral-Nemo-12B |
|--------|-----------|-----------------|
| Cohen's d(A,B) | 7.106 | 3.292 |
| Cohen's d(B,F) | 1.576 | 2.640 |
| H1 (Harm Concept) | ❌ Disconfirmed | ❌ Disconfirmed |
| H2 (Assist. Intent) | ❌ Disconfirmed | ❌ Disconfirmed |
| Refusal Layer | 22/32 | 28/40 |
| Cross-Model Pearson r | 0.710 | 0.710 |

**Conclusion:** The refusal direction is a compound feature that encodes both harm content awareness (moderate activation during passive reading) and production intent sensitivity (dramatically higher activation when asked to produce harmful content). This compound encoding is consistent across model scales.

---

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| **A** | Model is directly asked to produce harmful content |
| **B** | Model passively reads harmful content |
| **C** | Model critiques the writing style of harmful content |
| **D** | Model translates harmful content into French |
| **E** | Model writes fiction involving harmful scenarios |
| **F** | Model processes completely benign content (control) |

---

## Repository Structure

```
├── run_all_final.py              # Main pipeline orchestrator (Phases 0-6)
├── mini_capability_test.py       # Supplementary capability preservation test
├── q2_experiment/
│   ├── data/
│   │   ├── raw/                  # Raw source datasets (AdvBench, Alpaca)
│   │   ├── interim/              # Intermediate processing artifacts
│   │   └── processed/            # Final experiment set, train/test splits
│   ├── directions/               # Extracted refusal direction vectors (.pt)
│   ├── activations/              # Per-sample, per-condition activation data
│   │   ├── mistral_7b/
│   │   │   └── sample_*/
│   │   │       └── condition_*/
│   │   │           ├── model_response.txt
│   │   │           ├── cosine_all_layers_refusal.npy
│   │   │           └── last_token_all_layers.pt
│   │   └── mistral_nemo_12b/
│   ├── results/
│   │   ├── mistral_7b/           # Statistics, plots, LlamaGuard decisions
│   │   ├── mistral_nemo_12b/
│   │   └── cross_model/          # Cross-model alignment analysis
│   ├── output/
│   │   ├── paper_q2_refusal_direction.pdf   # Final research report
│   │   ├── paper_q2_refusal_direction.tex   # LaTeX source
│   │   └── result.md                         # Summary of results
│   └── logs/
│       ├── run_log.txt           # Full pipeline execution log
│       ├── phase_checkpoint.json # Checkpoint/recovery state
│       └── backup_*.tar.gz      # Hourly backups
```

---

## Pipeline Architecture

The experiment runs as a fully automated, checkpoint-resumable pipeline:

```
Phase 0: Synthetic Data Generation
    ├── Load AdvBench (520 harmful prompts) + Alpaca (2000 harmless prompts)
    ├── Extract refusal direction from Qwen model
    ├── Ablate Qwen and generate harmful completions
    └── Filter with LlamaGuard-2 → 359 verified pairs
         │
Phase 1: Train/Test Split
    └── test=50, train=154, experiment=155
         │
Phase 2: Direction Extraction (per model)
    ├── Difference-in-means at last input token
    ├── Optimal layer selection
    └── Behavioral validation (Ablation ASR + Addition Rate)
         │
Phase 3: Core Experiment (per model)
    ├── 6 conditions × 155 prompts = 930 measurements
    ├── Layer-wise activation recording
    └── Model response generation
         │
Phase 3.5: LlamaGuard Classification
    └── Classify all 930 responses per model
         │
Phase 4: Statistical Analysis
    ├── Cohen's d effect sizes
    ├── Bootstrap confidence intervals
    ├── Hypothesis testing
    └── Plot generation
         │
Phase 5: Cross-Model Alignment
    └── Pearson correlation across models
         │
Phase 6: Paper Generation
    └── LaTeX → PDF compilation
```

---

## Models Used

| Model | Parameters | Quantization | HuggingFace ID |
|-------|-----------|-------------|----------------|
| Mistral-7B | 7.24B | 4-bit NF4 | `mistralai/Mistral-7B-Instruct-v0.3` |
| Mistral-Nemo-12B | 12.2B | 4-bit NF4 | `mistralai/Mistral-Nemo-Instruct-2407` |
| Qwen (generation only) | 1.5B | 4-bit NF4 | `Qwen/Qwen2.5-1.5B-Instruct` |
| LlamaGuard-2 (classifier) | 8B | 4-bit NF4 | `meta-llama/Llama-Guard-2-8B` |

---

## Setup & Requirements

### Prerequisites
- Python 3.10+
- CUDA-capable GPU with ≥16GB VRAM
- Conda environment

### Installation
```bash
conda create -n q2_env python=3.10
conda activate q2_env
pip install torch transformers accelerate bitsandbytes scipy matplotlib numpy tqdm huggingface_hub
```

### Running the Pipeline
```bash
python run_all_final.py
```

The pipeline is fully checkpoint-resumable. If interrupted, simply re-run the same command and it will resume from the last completed phase.

---

## Results Visualization

### Cosine Similarity Across Conditions
The refusal direction shows a clear activation gradient: **A ≫ E > B > D ≈ C > F**

### Cross-Model Alignment
Strong behavioral consistency across model scales (Pearson r = 0.710), confirming that the compound feature encoding is architecture-independent.

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{sai2026refusal,
  title={Does the Refusal Direction Encode Harm or Production Intent? Evidence from the Mistral Model Family},
  author={Venkata Sai and Urja},
  year={2026},
  institution={IIT Patna, WCR Lab},
  supervisor={Prof. Preetam Kumar},
  guidance={Shambavi}
}
```

---

## References

1. Arditi, A., et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* arXiv:2406.11717.
2. Zou, A., et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043.
3. Jiang, A.Q., et al. (2023). *Mistral 7B.* arXiv:2310.06825.

---

## License

This project is for academic research purposes under the supervision of Prof. Preetam Kumar at IIT Patna.

---

*Built with ❤️ at WCR Lab, IIT Patna*

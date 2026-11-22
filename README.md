---
title: DepShield AI â€” 3D Supply-Chain Defense Studio
emoji: ðŸ›¡ï¸
colorFrom: purple
colorTo: green
sdk: gradio
app_file: app.py
app_port: 7860
pinned: false
---

# DepShield AI â€” Zero-Execution Supply-Chain Firewall for AI Coding Agents

**DepShield AI** is a multi-stage, zero-execution supply-chain firewall that intercepts malicious PyPI packages, AI-hallucinated dependencies (*slopsquatting*), typosquatting, and prompt-injection payloads **before** `pip install` ever runs.

> **Verified Benchmark Summary:** On a strictly held-out evaluation split of **28 malicious** and **21 clean/hard-negative** packages (`N=113` total held-out corpus evaluated in offline registry mode with zero PyPI-404 floor inflation), DepShield achieved **100.00% recall** (**95% CI: 87.94%â€“100.0%**) at **38.10% FPR** (**95% CI: 20.75%â€“59.12%**), catching **15/15** unseen spoofed-age evasion payloads.

---

## Architecture Overview

```text
  [ VS Code Copilot Agent (MCP) ]   [ CLI: depshield install ]   [ 3D WebGL Cyber Studio ]
                 \                            |                            /
                  v                           v                           v
        +---------------------------------------------------------------------------+
        |                    DepShield AI Ensemble Orchestrator                     |
        +---------------------------------------------------------------------------+
          |                   |                    |                    |
          v                   v                    v                    v
   [ Stage 1: AST ]   [ Stage 2A: RAG ]   [ Stage 2B: PyTorch ] [ Stage 2C: Modal ]
   Safe syntax tree   ChromaDB threat     TabularRiskNet        Fine-tuned QLoRA
   + Live PyPI 404    vector memory +     neural classifier     Qwen2.5-3B GPU
   + Allowlist/Index  LLM Judge           (age, entropy, AST)   code specialist
          \                   |                    |                    /
           +------------------+--------------------+-------------------+
                                          |
                                          v
                   Calibrated Verdict: ALLOW / WARN / BLOCK (>= 0.55)
                     + Safe Lockfile + Pushover Mobile Alert


# 🗂 Project Session Progress
> Last updated: October 25, 2026 — 01:23 IST
> Conversation ID: 6fe876e4-4924-467b-869d-304e276dde83

---

## What This Project Is
QLoRA-Tuned Llama 3.1 8B for Document Summarization.
- Fine-tuned on CNN/DailyMail (50K docs) with QLoRA (NF4 4-bit, rank=16)
- AWQ 4-bit quantization via autoawq
- FastAPI inference backend with vLLM + Redis caching
- Gradio demo UI
- Docker + Prometheus + Grafana stack

---

## COMPLETED (All Sessions)

### Session 1 — CI/CD & Repo Hygiene
- [x] Removed GITHUB/ subfolder (was misplaced)
- [x] Moved .gitignore to project root
- [x] Moved LICENSE to project root
- [x] Fixed pyproject.toml license path
- [x] Created .github/workflows/ci.yml (lint + test + coverage)
- [x] Added CI badge to README.md

### Session 2 — Code Fixes (Oct 25 2026)
- [x] Fix #1 — inference/main.py batch endpoint: replaced sequential for-loop with asyncio.gather()
- [x] Fix #2a — Deleted setup.py (was duplicating pyproject.toml, wrong version 1.0.0 vs 2.0.0)
- [x] Fix #2b — Added [tool.ruff] config to pyproject.toml (line-length=100, E/F/W/I rules)
- [x] Cleaned stale "setup.py" from [tool.coverage.run] omit list

---

## STILL TO DO (Priority Order)

### #3 — Grafana Provisioning Directory (HIGH — Docker stack broken)
docker-compose.yml mounts ./grafana/provisioning but dir does not exist.
Create:
  docker/grafana/provisioning/datasources/prometheus.yaml
  docker/grafana/provisioning/dashboards/dashboard.yaml
  docker/grafana/provisioning/dashboards/qlora_api.json

### #4 — Commit results/ folder (CRITICAL — credibility)
All ROUGE scores hardcoded in Python dicts — not real output.
Need: benchmark_report.json, rouge_benchmark.png, ablation_results.csv,
      sample_predictions.jsonl, training_loss_curve.png

### #5 — CHANGELOG.md + CONTRIBUTING.md + MODEL_CARD.md (MEDIUM)

### #6 — Add streaming endpoint test in tests/test_api.py (MEDIUM)

### #7 — Raise CI coverage threshold from 30% to 60%+ (MEDIUM)
  Edit .github/workflows/ci.yml line 119

### #8 — README polish: GIF, Quick Start, Known Limitations, cost table (MEDIUM)

### #9 — Push LoRA adapter to HuggingFace Hub (LOW-MEDIUM)

### #10 — Remove project_gaps_checklist.md from root before final push

---

## How to Resume
Open Antigravity IDE and say:
"I am back. Read .notes/session_progress.md and continue from #3 (Grafana provisioning directory)."

Or reference conversation ID: 6fe876e4-4924-467b-869d-304e276dde83

---

## Quick Commands
pytest -m "not gpu" -v
ruff check . --output-format=github
uvicorn inference.main:app --host 0.0.0.0 --port 8000 --workers 1
cd docker && docker compose up -d
python demo/app.py

# MemForge

Forge high-quality atomic memories from raw conversations.

Ingestion-first memory layer for AI agents — targeting LoCoMo SOTA with flexible multi-provider support.

## Quick Start

```bash
pip install -e ".[eval]"
export OPENROUTER_API_KEY=sk-or-...
python -m eval.locomo.run_eval --max-conversations 1
```

## Configuration

Edit `memforge.yaml` — see `examples/` for provider-specific configs.

# P05 — LLMOps with Ollama

Complete runnable demonstration using **local open models via Ollama**.

Demonstrates:
- Cost attribution (team + workload)
- Caching on stable prefixes (exact-match reuse)
- Hard budget + depth circuit breaker
- Cache-hit rate, savings, cap trips

## Prerequisites

1. Install [Ollama](https://ollama.com)
2. Pull a model:
   ```bash
   ollama pull llama3.2
   # or ollama pull qwen2.5:7b
   ```
3. Make sure Ollama is running (`ollama serve` or the desktop app)

## Quick Start

```bash
cd p05-llmops-agent
cp .env.example .env
# Edit OLLAMA_MODEL if you want a different model

pip install -r requirements.txt
python demo.py
```

Optional dashboard:
```bash
streamlit run dashboard.py
```

## Notes

- Caching is simulated with an in-memory exact-match cache (good for teaching).
- Token counts come from Ollama’s `prompt_eval_count` / `eval_count`.
- Pricing is simulated so you can still see “savings” numbers. Set prices to 0 in `.env` if you want pure free mode.
- The circuit breaker still works the same way.

## Primary metrics

- Cache-hit rate
- Savings versus non-cached baseline
- Attribution coverage (100%)
- Cap trips

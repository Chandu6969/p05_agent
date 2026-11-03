"""
P05 LLMOps Gateway — Ollama version
- Cost attribution by team + workload
- Simulated context caching (exact prompt match reuse)
- Hard budget cap + depth circuit breaker
- Metrics: cache-hit rate, savings, attribution coverage, cap trips
"""

import os
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

import ollama
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RequestMetrics:
    request_id: str
    team: str
    workload: str
    timestamp: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
    depth: int = 1
    stopped_by_breaker: bool = False
    reason: str = ""


@dataclass
class GatewayStats:
    total_requests: int = 0
    cache_hits: int = 0
    total_input_tokens: int = 0
    total_cached_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0
    cap_trips: int = 0
    requests: list = field(default_factory=list)

    @property
    def cache_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100

    @property
    def savings_usd(self) -> float:
        return max(0.0, self.baseline_cost_usd - self.total_cost_usd)

    @property
    def savings_percent(self) -> float:
        if self.baseline_cost_usd == 0:
            return 0.0
        return (self.savings_usd / self.baseline_cost_usd) * 100


class LLMOpsGateway:
    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.input_price = float(os.getenv("INPUT_PRICE_PER_MILLION", 0.10))
        self.output_price = float(os.getenv("OUTPUT_PRICE_PER_MILLION", 0.30))
        self.cached_price = float(os.getenv("CACHED_INPUT_PRICE_PER_MILLION", 0.01))

        self.team_budget = float(os.getenv("TEAM_BUDGET_USD", 0.50))
        self.max_depth = int(os.getenv("MAX_DEPTH", 8))

        self.stats = GatewayStats()
        self.team_spend: Dict[str, float] = {}
        self.cache: Dict[str, dict] = {}  # simple in-memory prompt cache
        self.stable_context = ""

        self._load_stable_context()
        print(f"[Gateway] Using Ollama model: {self.model}")
        print(f"[Gateway] Loaded stable context ({len(self.stable_context):,} chars)")

    def _load_stable_context(self):
        path = os.path.join(os.path.dirname(__file__), "data", "stable_context.txt")
        with open(path, "r", encoding="utf-8") as f:
            self.stable_context = f.read()

    def _make_cache_key(self, prompt: str) -> str:
        raw = self.stable_context + "||" + prompt
        return hashlib.sha256(raw.encode()).hexdigest()

    def _calculate_cost(self, input_tokens: int, output_tokens: int, cached_tokens: int) -> tuple[float, float]:
        non_cached = max(0, input_tokens - cached_tokens)
        actual = (
            (non_cached / 1_000_000) * self.input_price
            + (cached_tokens / 1_000_000) * self.cached_price
            + (output_tokens / 1_000_000) * self.output_price
        )
        baseline = (
            (input_tokens / 1_000_000) * self.input_price
            + (output_tokens / 1_000_000) * self.output_price
        )
        return actual, baseline

    def _check_breaker(self, team: str, depth: int) -> Optional[str]:
        current = self.team_spend.get(team, 0.0)
        if current >= self.team_budget:
            return f"Budget cap reached for team '{team}' (${current:.4f} >= ${self.team_budget:.2f})"
        if depth > self.max_depth:
            return f"Maximum agent depth exceeded ({depth} > {self.max_depth})"
        return None

    def call(
        self,
        prompt: str,
        team: str = "engineering",
        workload: str = "research",
        depth: int = 1,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat()

        breaker_reason = self._check_breaker(team, depth)
        if breaker_reason:
            self.stats.cap_trips += 1
            metric = RequestMetrics(
                request_id=request_id,
                team=team,
                workload=workload,
                timestamp=timestamp,
                depth=depth,
                stopped_by_breaker=True,
                reason=breaker_reason,
            )
            self.stats.requests.append(metric)
            self.stats.total_requests += 1
            return {
                "request_id": request_id,
                "stopped": True,
                "reason": breaker_reason,
                "text": f"[GATEWAY STOPPED] {breaker_reason}",
            }

        cache_key = self._make_cache_key(prompt)
        cache_hit = False
        cached_tokens = 0

        if use_cache and cache_key in self.cache:
            # Simulated cache hit
            cached = self.cache[cache_key]
            text = cached["text"]
            input_tokens = cached["input_tokens"]
            output_tokens = cached["output_tokens"]
            cached_tokens = input_tokens  # full input treated as cached
            cache_hit = True
            print(f"[Cache HIT] {request_id}")
        else:
            # Real Ollama call
            messages = [
                {"role": "system", "content": self.stable_context},
                {"role": "user", "content": prompt},
            ]
            try:
                response = ollama.chat(model=self.model, messages=messages)
                text = response["message"]["content"]

                # Token counts from Ollama
                input_tokens = response.get("prompt_eval_count", 0) or 0
                output_tokens = response.get("eval_count", 0) or 0

                # Store in cache
                self.cache[cache_key] = {
                    "text": text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
                print(f"[Cache MISS] {request_id} → calling Ollama")
            except Exception as e:
                return {
                    "request_id": request_id,
                    "stopped": True,
                    "reason": str(e),
                    "text": f"[ERROR] {e}",
                }

        actual_cost, baseline_cost = self._calculate_cost(input_tokens, output_tokens, cached_tokens)

        self.stats.total_requests += 1
        self.stats.total_input_tokens += input_tokens
        self.stats.total_output_tokens += output_tokens
        self.stats.total_cached_tokens += cached_tokens
        self.stats.total_cost_usd += actual_cost
        self.stats.baseline_cost_usd += baseline_cost
        if cache_hit:
            self.stats.cache_hits += 1

        self.team_spend[team] = self.team_spend.get(team, 0.0) + actual_cost

        metric = RequestMetrics(
            request_id=request_id,
            team=team,
            workload=workload,
            timestamp=timestamp,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=actual_cost,
            cache_hit=cache_hit,
            depth=depth,
        )
        self.stats.requests.append(metric)

        return {
            "request_id": request_id,
            "stopped": False,
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": actual_cost,
            "baseline_cost_usd": baseline_cost,
            "cache_hit": cache_hit,
            "team_spend_so_far": self.team_spend.get(team, 0.0),
        }

    def get_dashboard(self) -> dict:
        s = self.stats
        return {
            "total_requests": s.total_requests,
            "cache_hit_rate_pct": round(s.cache_hit_rate, 1),
            "total_cost_usd": round(s.total_cost_usd, 6),
            "baseline_cost_usd": round(s.baseline_cost_usd, 6),
            "savings_usd": round(s.savings_usd, 6),
            "savings_percent": round(s.savings_percent, 1),
            "cap_trips": s.cap_trips,
            "attribution_coverage_pct": 100.0 if s.total_requests > 0 else 0.0,
            "team_spend": dict(self.team_spend),
        }

    def print_dashboard(self):
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()
        d = self.get_dashboard()

        table = Table(title="P05 Unit Economics Dashboard (Ollama)", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Requests", str(d["total_requests"]))
        table.add_row("Cache-Hit Rate", f"{d['cache_hit_rate_pct']}%")
        table.add_row("Actual Cost", f"${d['total_cost_usd']:.6f}")
        table.add_row("Baseline Cost (no cache)", f"${d['baseline_cost_usd']:.6f}")
        table.add_row("Savings", f"${d['savings_usd']:.6f} ({d['savings_percent']}%)")
        table.add_row("Cap Trips", str(d["cap_trips"]))
        table.add_row("Attribution Coverage", f"{d['attribution_coverage_pct']}%")

        console.print(Panel(table, title="LLMOps Gateway Metrics", border_style="blue"))

        if d["team_spend"]:
            console.print("\n[bold]Team Spend:[/bold]")
            for team, spend in d["team_spend"].items():
                console.print(f"  {team}: ${spend:.6f}")

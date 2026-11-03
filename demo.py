"""
P05 Demonstration — Ollama version
"""

import time
from gateway import LLMOpsGateway
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    console.print(Panel.fit(
        "[bold cyan]P05 LLMOps Demo (Ollama)[/bold cyan]\n"
        "Caching + FinOps + Circuit Breaker",
        border_style="cyan"
    ))

    gateway = LLMOpsGateway()

    # Phase 1: Stable workload (should hit cache on repeats)
    console.print("\n[bold yellow]Phase 1: Stable workload (cache expected)[/bold yellow]")

    questions = [
        "What is the caching policy for documents longer than 4000 tokens?",
        "Explain the circuit breaker rules.",
        "What is the tagging convention?",
        "How does showback/chargeback work?",
        "What is the target cache-hit rate?",
        # Repeat one to force a cache hit
        "What is the caching policy for documents longer than 4000 tokens?",
    ]

    for i, q in enumerate(questions, 1):
        console.print(f"\n[bold]Request {i}:[/bold] {q}")
        result = gateway.call(
            prompt=q,
            team="engineering",
            workload="research",
            use_cache=True,
        )
        if result.get("stopped"):
            console.print(f"[red]{result['text']}[/red]")
        else:
            console.print(f"[green]Cache hit: {result['cache_hit']} | "
                          f"Cost: ${result['cost_usd']:.6f} | "
                          f"Cached tokens: {result['cached_tokens']}[/green]")
            text = result["text"][:180] + "..." if len(result["text"]) > 180 else result["text"]
            console.print(f"Answer: {text}")
        time.sleep(0.3)

    # Phase 2: Runaway loop
    console.print("\n[bold red]Phase 2: Runaway agent loop (should be stopped by breaker)[/bold red]")

    depth = 1
    while depth <= 12:
        console.print(f"\n[bold]Loop step {depth}[/bold]")
        result = gateway.call(
            prompt=f"Continue the previous analysis. Step {depth}. Keep going.",
            team="engineering",
            workload="research",
            depth=depth,
            use_cache=True,
        )
        if result.get("stopped"):
            console.print(f"[bold red]STOPPED: {result['reason']}[/bold red]")
            break
        else:
            console.print(f"Cost so far for team: ${result['team_spend_so_far']:.6f}")
        depth += 1
        time.sleep(0.2)

    console.print("\n")
    gateway.print_dashboard()

    console.print("\n[bold green]Demo complete.[/bold green]")
    console.print("Key takeaways:")
    console.print("• Stable prefix was cached → lower cost on repeats")
    console.print("• Runaway loop was stopped by budget/depth circuit breaker")
    console.print("• Every request was attributed to team + workload")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Token Inspector CLI

Features:
- Inspect token IDs and decoded chunks
- Make whitespace visible
- Show useful stats
- Compare two inputs
- Optional rough cost estimate
- Rich-colored terminal output

Examples:
    python token_inspector.py "hello world"
    python token_inspector.py "hello" " hello"
    python token_inspector.py --encoding cl100k_base "Kubernetes is powerful"
    python token_inspector.py --price-per-1k 0.005 "A longer prompt goes here"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean
from typing import List, Sequence

import tiktoken
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


VISIBLE_WHITESPACE_MAP = {
    " ": "␠",
    "\n": "↵\n",
    "\t": "⇥",
    "\r": "␍",
}


@dataclass
class TokenInfo:
    index: int
    token_id: int
    decoded: str
    visible: str
    char_length: int


@dataclass
class AnalysisResult:
    text: str
    encoding_name: str
    token_ids: List[int]
    tokens: List[TokenInfo]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def avg_token_length(self) -> float:
        if not self.tokens:
            return 0.0
        return mean(t.char_length for t in self.tokens)

    @property
    def longest_tokens(self) -> List[TokenInfo]:
        if not self.tokens:
            return []
        max_len = max(t.char_length for t in self.tokens)
        return [t for t in self.tokens if t.char_length == max_len]


def make_whitespace_visible(text: str) -> str:
    out = text
    for raw, visible in VISIBLE_WHITESPACE_MAP.items():
        out = out.replace(raw, visible)
    return out


def analyse_text(text: str, encoding_name: str) -> AnalysisResult:
    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(text)

    tokens: List[TokenInfo] = []
    for idx, token_id in enumerate(token_ids, start=1):
        decoded = enc.decode([token_id])
        tokens.append(
            TokenInfo(
                index=idx,
                token_id=token_id,
                decoded=decoded,
                visible=make_whitespace_visible(decoded),
                char_length=len(decoded),
            )
        )

    return AnalysisResult(
        text=text,
        encoding_name=encoding_name,
        token_ids=token_ids,
        tokens=tokens,
    )


def build_token_table(result: AnalysisResult) -> Table:
    table = Table(title="Token Breakdown", show_lines=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Token ID", justify="right", style="magenta")
    table.add_column("Decoded", style="green")
    table.add_column("Chars", justify="right", style="yellow")

    for token in result.tokens:
        visible = token.visible if token.visible else "∅"
        table.add_row(
            str(token.index),
            str(token.token_id),
            repr(visible),
            str(token.char_length),
        )

    return table


def build_stats_table(result: AnalysisResult, price_per_1k: float | None) -> Table:
    table = Table(title="Stats", show_header=False)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Encoding", result.encoding_name)
    table.add_row("Characters", str(result.char_count))
    table.add_row("Tokens", str(result.token_count))
    table.add_row("Avg chars/token", f"{result.avg_token_length:.2f}")

    if result.longest_tokens:
        longest_visible = ", ".join(
            repr(t.visible if t.visible else "∅") for t in result.longest_tokens[:5]
        )
        table.add_row(
            "Longest token(s)",
            f"{longest_visible} ({result.longest_tokens[0].char_length} chars)",
        )

    if price_per_1k is not None:
        estimated_cost = (result.token_count / 1000) * price_per_1k
        table.add_row("Est. cost", f"${estimated_cost:.6f}")

    return table


def show_single_analysis(result: AnalysisResult, price_per_1k: float | None) -> None:
    console.print(
        Panel.fit(
            f"[bold]Input[/bold]\n{repr(make_whitespace_visible(result.text))}",
            title="Text",
        )
    )
    console.print(build_stats_table(result, price_per_1k))
    console.print(build_token_table(result))


def compare_results(a: AnalysisResult, b: AnalysisResult) -> None:
    table = Table(title="Comparison", show_header=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Input A", style="green")
    table.add_column("Input B", style="green")
    table.add_column("Difference (B - A)", style="yellow")

    table.add_row("Characters", str(a.char_count), str(b.char_count), str(b.char_count - a.char_count))
    table.add_row("Tokens", str(a.token_count), str(b.token_count), str(b.token_count - a.token_count))
    table.add_row(
        "Avg chars/token",
        f"{a.avg_token_length:.2f}",
        f"{b.avg_token_length:.2f}",
        f"{(b.avg_token_length - a.avg_token_length):.2f}",
    )

    console.print(
        Panel.fit(
            f"[bold]Input A[/bold]\n{repr(make_whitespace_visible(a.text))}\n\n"
            f"[bold]Input B[/bold]\n{repr(make_whitespace_visible(b.text))}",
            title="Compared Inputs",
        )
    )
    console.print(table)

    ids_a = a.token_ids
    ids_b = b.token_ids
    min_len = min(len(ids_a), len(ids_b))

    first_diff = None
    for i in range(min_len):
        if ids_a[i] != ids_b[i]:
            first_diff = i
            break

    if first_diff is None and len(ids_a) != len(ids_b):
        first_diff = min_len

    if first_diff is None:
        console.print("[bold green]Token sequences are identical.[/bold green]")
        return

    console.print(f"[bold yellow]First difference at token position:[/bold yellow] {first_diff + 1}")

    diff_table = Table(title="Local Difference View")
    diff_table.add_column("Pos", justify="right", style="cyan")
    diff_table.add_column("A ID", justify="right", style="magenta")
    diff_table.add_column("A Decoded", style="green")
    diff_table.add_column("B ID", justify="right", style="magenta")
    diff_table.add_column("B Decoded", style="green")

    start = max(0, first_diff - 3)
    end = first_diff + 4

    for pos in range(start, end):
        a_id = ids_a[pos] if pos < len(ids_a) else None
        b_id = ids_b[pos] if pos < len(ids_b) else None

        a_decoded = make_whitespace_visible(a.tokens[pos].decoded) if pos < len(a.tokens) else "∅"
        b_decoded = make_whitespace_visible(b.tokens[pos].decoded) if pos < len(b.tokens) else "∅"

        diff_table.add_row(
            str(pos + 1),
            str(a_id) if a_id is not None else "∅",
            repr(a_decoded),
            str(b_id) if b_id is not None else "∅",
            repr(b_decoded),
        )

    console.print(diff_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and compare LLM tokenization.")
    parser.add_argument(
        "texts",
        nargs="+",
        help="One or two text inputs. If two are supplied, comparison mode is used.",
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="tiktoken encoding to use (default: cl100k_base)",
    )
    parser.add_argument(
        "--price-per-1k",
        type=float,
        default=None,
        help="Optional rough token price per 1,000 tokens for estimating cost.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if len(args.texts) not in {1, 2}:
        console.print("[bold red]Provide one or two text inputs only.[/bold red]")
        return 1

    try:
        if len(args.texts) == 1:
            result = analyse_text(args.texts[0], args.encoding)
            show_single_analysis(result, args.price_per_1k)
        else:
            a = analyse_text(args.texts[0], args.encoding)
            b = analyse_text(args.texts[1], args.encoding)
            compare_results(a, b)
    except ValueError as exc:
        console.print(f"[bold red]Encoding error:[/bold red] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())    

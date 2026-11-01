"""`hkn` command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from hkn.config import get_settings

app = typer.Typer(
    help="Healthcare Knowledge Navigator", no_args_is_help=True, pretty_exceptions_enable=False
)
ingest_app = typer.Typer(help="Build the corpus", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation harness", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(eval_app, name="eval")
console = Console()


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@ingest_app.command("seed")
def ingest_seed(
    only: list[str] = typer.Option(None, "--only", help="source_key(s) to ingest"),  # noqa: B008
    refresh: bool = typer.Option(False, help="re-download even if cached"),
    force: bool = typer.Option(False, help="re-ingest even if unchanged"),
    no_embed: bool = typer.Option(False, help="skip embeddings (parse/store only)"),
) -> None:
    """Fetch, parse, chunk and embed the seed manifest."""
    from hkn.db import session_scope
    from hkn.ingest.embed import embed_passages
    from hkn.ingest.pipeline import ingest_spec
    from hkn.ingest.sources import load_manifest

    settings = get_settings()
    manifest = load_manifest(settings.manifest_path)
    specs = [s for s in manifest.documents if not only or s.source_key in only]
    if only and len(specs) != len(only):
        missing = set(only) - {s.source_key for s in specs}
        raise typer.BadParameter(f"unknown source_key(s): {sorted(missing)}")

    def embedder(texts: list[str]) -> list[list[float]]:
        return embed_passages(texts, show_progress=len(texts) > 200)

    table = Table("source", "status", "sections", "sentences", "chunks")
    for spec in specs:
        console.print(f"[bold]{spec.source_key}[/]  {spec.url}")
        with session_scope() as session:
            result = ingest_spec(
                session, spec, manifest, embedder=None if no_embed else embedder,
                refresh=refresh, force=force,
            )  # fmt: skip
        table.add_row(
            spec.source_key,
            result.status,
            str(result.sections),
            str(result.sentences),
            str(result.chunks),
        )
    console.print(table)


@ingest_app.command("pmc")
def ingest_pmc(
    per_topic: int = typer.Option(75, help="articles per topic (4 topics in the manifest)"),
    topic: list[str] = typer.Option(None, "--topic", help="specialty name(s) to ingest"),  # noqa: B008
    no_embed: bool = typer.Option(False),
) -> None:
    """Ingest CC BY / CC0 PubMed Central articles for the manifest's topics."""
    from hkn.db import session_scope
    from hkn.ingest.embed import embed_passages
    from hkn.ingest.pipeline import DocMeta, store_document
    from hkn.ingest.pmc import PmcArticle, fetch_articles
    from hkn.ingest.sources import load_manifest

    settings = get_settings()
    if not settings.ncbi_email:
        console.print(
            "[yellow]HKN_NCBI_EMAIL is not set; NCBI asks every tool to identify a contact address.[/]"
        )
    topics = [
        t
        for t in load_manifest(settings.manifest_path).pmc_topics
        if not topic or t["specialty"] in topic
    ]
    counts = {"ingested": 0, "unchanged": 0, "skipped": 0}
    for t in topics:
        console.print(f"[bold]{t['specialty']}[/]: {t['query']}")
        for item in fetch_articles(t["query"], limit=per_topic, settings=settings):
            if not isinstance(item, PmcArticle):
                counts["skipped"] += 1
                console.print(f"  skip {item[0]}: {item[1]}")
                continue
            p = item.parsed
            meta = DocMeta(
                source_key=f"pmc-{item.pmcid.lower()}",
                title=p.title or item.pmcid,
                org="PubMed Central",
                doc_type=p.doc_type or "other",
                url=f"https://pmc.ncbi.nlm.nih.gov/articles/{item.pmcid}/",
                license=p.license or "",
                specialty=[t["specialty"]],
                pub_year=p.pub_year,
                version=item.pmcid,
                authors=f"{p.authors}. {item.journal}." if item.journal else p.authors,
            )
            with session_scope() as session:
                result = store_document(
                    session, meta, p, sha256=item.sha256,
                    embedder=None if no_embed else embed_passages, settings=settings,
                )  # fmt: skip
            counts[result.status] += 1
            console.print(f"  {result.status} {item.pmcid} {p.doc_type}: {(p.title or '')[:70]}")
    console.print(counts)


@ingest_app.command("contextualize")
def ingest_contextualize(
    scope: str = typer.Option(
        "guidelines", help="guidelines (guidelines, protocols, labels) or all"
    ),
    limit: int = typer.Option(None, help="first N chunks only"),
    yes: bool = typer.Option(False, "--yes", help="submit the batch (paid)"),
    collect_results: bool = typer.Option(
        False, "--collect", help="wait for submitted batches and store results"
    ),
) -> None:
    """Claude-written chunk contexts via the Message Batches API (opt-in, cost-gated)."""
    from hkn.db import session_scope
    from hkn.ingest import contextualize as ctx
    from hkn.ingest.embed import embed_passages

    settings = get_settings()
    if collect_results:
        console.print(ctx.collect(embed_passages, settings))
        return
    if scope not in {"guidelines", "all"}:
        raise typer.BadParameter("scope must be 'guidelines' or 'all'")
    with session_scope() as session:
        requests = ctx.build_requests(session, scope=scope, limit=limit)
    if not requests:
        console.print("nothing to contextualize")
        return
    est = ctx.estimate(requests, settings)
    console.print(
        f"{len(requests)} chunks on {settings.context_model} via Batches (50% off): ~${est:.2f}"
    )
    if not yes:
        console.print(
            "re-run with --yes to submit; HKN_CONTEXT_MODEL=claude-haiku-5-5 is much cheaper"
        )
        return
    batch_id = ctx.submit(requests, settings)
    console.print(
        f"submitted batch {batch_id}; run `hkn ingest contextualize --collect` to store results"
    )


@app.command("search")
def search_cmd(
    question: str,
    no_plan: bool = typer.Option(False, help="skip the Claude query plan (free, offline)"),
    no_rerank: bool = typer.Option(False),
    top_k: int = typer.Option(8),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Run retrieval only and print the ranked chunks."""
    from hkn.db import session_scope
    from hkn.retrieval.pipeline import RetrievalConfig, retrieve
    from hkn.retrieval.plan_query import plan_query

    config = RetrievalConfig.from_settings().with_(
        use_query_plan=not no_plan, rerank=not no_rerank, top_k=top_k
    )
    plan, plan_error = (None, None) if no_plan else plan_query(question)
    if plan_error:
        console.print(f"[yellow]query plan unavailable: {plan_error}[/]")
    with session_scope() as session:
        result = retrieve(session, question, plan=plan, config=config)
    if as_json:
        console.print_json(json.dumps(result.trace()))
        return
    table = Table("#", "rerank", "rrf", "document", "heading", "text")
    for i, c in enumerate(result.candidates, 1):
        table.add_row(
            str(i),
            "-" if c.rerank_score is None else f"{c.rerank_score:.2f}",
            f"{c.rrf_score:.4f}",
            f"{c.org} {c.pub_year or ''}",
            " › ".join(c.heading_path)[-60:],
            c.text[:120],
        )
    console.print(table)
    console.print(f"timings: {result.timings_ms}  pool: {result.pool_size}")


@app.command("ask")
def ask_cmd(
    question: str,
    doc_type: list[str] = typer.Option(None, "--doc-type", help="hard filter, repeatable"),  # noqa: B008
    org: list[str] = typer.Option(None, "--org", help="hard filter, repeatable"),  # noqa: B008
    no_plan: bool = typer.Option(False, help="skip the query plan call"),
    as_json: bool = typer.Option(False, "--json", help="print the raw event list"),
) -> None:
    """Answer a question with citations and a confidence score (calls Claude)."""
    from hkn.ask import ask
    from hkn.retrieval.pipeline import RetrievalConfig
    from hkn.retrieval.search import Filters

    config = RetrievalConfig.from_settings().with_(use_query_plan=not no_plan)
    result = ask(
        question, filters=Filters(doc_types=doc_type or None, orgs=org or None), config=config
    )
    if as_json:
        console.print_json(json.dumps(result.events, default=str))
        return
    if result.status != "answered":
        console.print(f"[bold yellow]{result.status.upper()}[/]: {result.reason}")
        return
    for block in result.answer.blocks:
        marks = sorted({c.source_index + 1 for c in block.citations})
        console.print(
            block.text + ("" if not marks else " " + "".join(f"[cyan][{m}][/]" for m in marks)),
            end="",
        )
    console.print("\n")
    cited = sorted({c.source_index for c in result.answer.citations})
    for src in result.sources:
        if src.index in cited:
            console.print(f"[cyan][{src.index + 1}][/] {src.title}\n    {src.source_url}")
    rel = result.reliability
    if rel is not None:
        tag = "" if rel.calibrated else " (uncalibrated prior)"
        console.print(
            f"\n[bold]Reliability:[/] {rel.band} {rel.score:.2f}{tag}  features={rel.features}"
        )
        for reason in rel.reasons:
            console.print(f"  - {reason}")
    if result.evidence:
        console.print(
            f"[bold]Evidence strength:[/] {result.evidence['rating']} — {'; '.join(result.evidence['reasons'])}"
        )


@app.command("record-fixture")
def record_fixture(
    question: str = typer.Argument(
        "What is the recommended treatment for uncomplicated gonorrhea?"
    ),
    name: str = typer.Option("answer_stream_recorded.json"),
) -> None:
    """Record one real answer stream (raw events + sources) as a test fixture (calls Claude)."""
    from hkn.ask import prepare, sources_payload
    from hkn.generation.answer import build_request
    from hkn.llm import get_client

    p = prepare(question)
    if not p.gate.passed:
        console.print(f"[red]retrieval gate failed:[/] {p.gate.reason}")
        raise typer.Exit(1)
    events = []
    with get_client().beta.messages.stream(**build_request(question, p.sources)) as stream:
        for event in stream:
            if event.type in {
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
            }:
                events.append(event.to_dict())
        final = stream.get_final_message().to_dict()
    path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / name
    path.write_text(
        json.dumps(
            {
                "question": question,
                "sources": sources_payload(p.sources),
                "events": events,
                "final": final,
            },
            indent=1,
        )
    )
    console.print(f"recorded {len(events)} events to {path}")


if __name__ == "__main__":
    app()

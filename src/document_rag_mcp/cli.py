import asyncio
import os
from pathlib import Path
import click


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    envvar="DOCRAG_CONFIG",
    help="Path to the YAML configuration file.",
)
@click.option(
    "--chunking-model",
    envvar="DOCRAG_CHUNKING_MODEL",
    help="Override the local model used for semantic chunking boundary detection.",
)
@click.pass_context
def main(ctx: click.Context, config: Path | None, chunking_model: str | None) -> None:
    """RAG MCP Server CLI — Manage document indexing and semantic search server."""
    if config:
        os.environ["DOCRAG_CONFIG"] = str(config.resolve())
    if chunking_model:
        os.environ["DOCRAG_CHUNKING_MODEL"] = chunking_model


@main.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"]),
    default="stdio",
    help="Transport protocol to run the server on (stdio or http/sse).",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind the HTTP server to (http transport only).",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to run the HTTP server on (http transport only).",
)
def serve(transport: str, host: str, port: int) -> None:
    """Start the Model Context Protocol (MCP) server."""
    # Lazy import to ensure CLI overrides are set in environment first
    import document_rag_mcp.server as mcp_server

    transport_mcp = "sse" if transport == "http" else "stdio"
    click.echo(f"Starting document-rag-mcp server on {transport} transport...")
    if transport == "http":
        click.echo(f"Binding to http://{host}:{port}")

    mcp_server.mcp.run(transport=transport_mcp, host=host, port=port)


@main.command()
@click.argument("query")
@click.option(
    "--collection",
    "-c",
    default=None,
    help="Filter search results by a specific collection.",
)
@click.option(
    "--top-k",
    "-k",
    default=5,
    type=int,
    help="Number of results to return.",
)
def search(query: str, collection: str | None, top_k: int) -> None:
    """Run a semantic search against the indexed collections."""
    import document_rag_mcp.server as mcp_server

    async def run_search():
        click.echo(f"Searching for: '{query}'...")
        results = await mcp_server.search_engine.search(
            query=query, collection_name=collection, top_k=top_k
        )
        if not results:
            click.echo("No matching documents found.")
            return

        for i, r in enumerate(results):
            click.echo(f"\n[{i+1}] Score: {r.score:.4f} | File: {r.metadata.file_path}")
            click.echo(f"Collection: {r.metadata.collection} | Page: {r.metadata.page_number or 1}")
            click.echo(f"Title: {r.metadata.title or 'N/A'} | Section: {r.metadata.section or 'N/A'}")
            click.echo("-" * 40)
            click.echo(r.text.strip())
            click.echo("=" * 60)

    asyncio.run(run_search())


@main.command()
@click.option(
    "--collection",
    "-c",
    default=None,
    help="Limit ingestion to a specific collection.",
)
def ingest(collection: str | None) -> None:
    """Recursively scan folders and index new/changed files immediately."""
    import document_rag_mcp.server as mcp_server

    async def run_ingest():
        click.echo("Starting one-shot ingestion scan...")
        res = await mcp_server.ingest_now(collection=collection)
        click.echo(res)

    asyncio.run(run_ingest())


@main.command()
def collections() -> None:
    """List all configured collections and their indexing status."""
    import document_rag_mcp.server as mcp_server

    click.echo("Configured Collections:")
    click.echo("=" * 60)
    for coll in mcp_server.config.collections:
        stats = mcp_server.vector_store.collection_stats(coll.name)
        paths_str = ", ".join(str(p) for p in coll.paths)
        click.echo(f"Name:          {coll.name}")
        click.echo(f"Paths:         {paths_str}")
        click.echo(f"Patterns:      {', '.join(coll.file_patterns)}")
        click.echo(f"Indexed Chunks: {stats['count']}")
        click.echo("-" * 60)


if __name__ == "__main__":
    main()

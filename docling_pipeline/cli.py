"""Command-line interface (CLI) for the Docling Hybrid Document Pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docling_pipeline.pipeline import HybridDocumentPipeline
from docling_pipeline.tools.mcp_tool import (
    create_mcp_server,
    inspect_excel_spreadsheet,
    triage_pdf_document,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> None:
    # Ensure UTF-8 output on Windows console to avoid charmap encoding errors with Vietnamese/Unicode
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Docling Hybrid Pipeline CLI: Convert documents, triage PDFs, inspect spreadsheets, or start MCP server."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        help="Path to the document to process (PDF, XLSX, DOCX, PPTX, CSV, TXT).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output markdown file (if omitted, prints to console).",
    )
    parser.add_argument(
        "--triage-only",
        action="store_true",
        help="Only run page-level triage on a PDF file.",
    )
    parser.add_argument(
        "--inspect-excel",
        action="store_true",
        help="Only run single-pass cell formula & value inspection on an Excel file.",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Start the FastMCP server for AI agents and LLM tool calls.",
    )
    parser.add_argument(
        "--vlm-url",
        help="URL of local or self-hosted VLM/vLLM server (e.g. http://localhost:8000/v1/chat/completions).",
    )
    parser.add_argument(
        "--vlm-model",
        help="Model name for local VLM/vLLM server (e.g. Qwen/Qwen2-VL-7B-Instruct).",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable VLM visual processing entirely (use clean image placeholders).",
    )
    parser.add_argument(
        "--concurrency",
        "--vlm-concurrency",
        type=int,
        default=None,
        dest="vlm_concurrency",
        help="Maximum concurrent async requests to VLM server (default: 1 or VLM_MAX_CONCURRENCY).",
    )
    parser.add_argument(
        "--timeout",
        "--vlm-timeout",
        type=float,
        default=None,
        dest="vlm_timeout",
        help="Timeout in seconds per VLM request (default: 60.0 or VLM_TIMEOUT).",
    )
    parser.add_argument(
        "--vlm-dim",
        "--vlm-max-dim",
        type=int,
        default=None,
        dest="vlm_max_dim",
        help="Maximum image dimension for vision model downscaling (default: 1024 or VLM_MAX_DIM).",
    )

    args = parser.parse_args()

    # 1. Mode MCP Server
    if args.mcp:
        print("Starting FastMCP server on stdio...")
        server = create_mcp_server()
        if server:
            server.run()
        else:
            print("Error: FastMCP is not available.")
            sys.exit(1)
        return

    if not args.file_path:
        parser.print_help()
        sys.exit(0)

    target_path = Path(args.file_path)
    if not target_path.exists():
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)

    # 2. Mode PDF Triage Only
    if args.triage_only:
        report = triage_pdf_document(str(target_path))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    # 3. Mode Excel Inspect Only
    if args.inspect_excel:
        inspection = inspect_excel_spreadsheet(str(target_path))
        print(json.dumps(inspection, indent=2, ensure_ascii=False))
        return

    # 4. Standard Hybrid Pipeline Processing
    print(f"[*] Processing document with Docling Hybrid Pipeline: {target_path.name}...")
    pipeline = HybridDocumentPipeline(
        vlm_server_url=args.vlm_url,
        vlm_model=args.vlm_model,
        vlm_concurrency=args.vlm_concurrency,
        vlm_timeout=args.vlm_timeout,
        vlm_max_dim=args.vlm_max_dim,
        enable_vlm=not args.no_vlm,
    )
    result = pipeline.process(target_path)

    print(f"[+] Completed in {result.processing_time_seconds}s (Format: {result.file_format.upper()})")

    if args.output:
        raw_out = str(args.output)
        out_path = Path(raw_out)
        if out_path.is_dir() or raw_out.endswith(("\\", "/")):
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{target_path.stem}.md"
            base_dir = out_path
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_file = out_path
            base_dir = out_path.parent
        out_file.write_text(result.markdown, encoding="utf-8")
        print(f"[+] Output saved to: {out_file.resolve()}")

        if result.extracted_images:
            for rel_img_path, img_bytes in result.extracted_images.items():
                img_dest = base_dir / rel_img_path
                img_dest.parent.mkdir(parents=True, exist_ok=True)
                img_dest.write_bytes(img_bytes)
                print(f"[+] Extracted image saved to: {img_dest.resolve()}")
    else:
        print("\n--- [RESULT MARKDOWN] ---\n")
        print(result.markdown)


if __name__ == "__main__":
    main()

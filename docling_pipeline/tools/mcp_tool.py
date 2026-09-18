"""FastMCP server and Python callable AI tool interface for document ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docling_pipeline.pipeline import HybridDocumentPipeline
from docling_pipeline.spreadsheet.excel_inspector import inspect_excel
from docling_pipeline.spreadsheet.formula_semantic import FormulaSemanticMapper
from docling_pipeline.triage.pdf_triage import triage_pdf

# Global pipeline instance
_pipeline = HybridDocumentPipeline()


def convert_document_to_markdown(
    file_path: str,
    enable_formula_semantics: bool = True,
) -> str:
    """Convert any document (PDF, DOCX, XLSX, PPTX, TXT, CSV) into clean RAG-ready Markdown.

    Features:
    - Native oMath2Latex preservation for Word/PPTX ($...$ LaTeX formulas)
    - Single-pass formula and cached-value extraction for Excel with fallback
    - Page-level adaptive triage for PDF (bypasses OCR for digital pages, triggers OCR for scans)
    - Safe VLM annotation hooks emitting Mermaid flowcharts and Markdown tables
    """
    path = Path(file_path)
    if not path.exists():
        return f"Error: Document not found at {file_path}"

    pipe = _pipeline if enable_formula_semantics else HybridDocumentPipeline(enable_formula_semantics=False)
    res = pipe.process(path)
    return res.markdown


def triage_pdf_document(file_path: str) -> dict[str, Any]:
    """Inspect a PDF page-by-page to detect digital vector text vs scanned pages.

    Filters out decorative icons (< 100x100 px) and identifies diagram/chart candidates for VLM.
    """
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    report = triage_pdf(path)
    return {
        "file_path": str(path),
        "total_pages": report.total_pages,
        "requires_ocr": report.requires_ocr,
        "has_vlm_candidates": report.has_vlm_candidates,
        "direct_parse_pages": report.direct_parse_pages,
        "ocr_pages": report.ocr_pages,
        "hybrid_pages": report.hybrid_pages,
        "pages": [
            {
                "page": p.page_number,
                "char_count": p.char_count,
                "triage_mode": p.triage_mode.value,
                "vlm_candidates": p.vlm_candidate_images_count,
                "decision": p.decision_reason,
            }
            for p in report.pages
        ],
    }


def inspect_excel_spreadsheet(file_path: str) -> dict[str, Any]:
    """Single-pass cell inspector for Excel spreadsheets.

    Bypasses data_only=True limitations, extracts formula (<f>) and value (<v>) pairs,
    provides fallback evaluation, and formats prompts for LLM semantic translation.
    """
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    res = inspect_excel(path)
    formula_summary = res.get_formula_summary()

    semantic_prompts = []
    for sheet in res.sheets:
        if sheet.formula_cells_count > 0:
            prompt_container = FormulaSemanticMapper.map_sheet_formulas(sheet)
            semantic_prompts.append(
                {
                    "sheet": sheet.name,
                    "prompt_en": prompt_container.to_llm_prompt("en"),
                    "prompt_vi": prompt_container.to_llm_prompt("vi"),
                }
            )

    return {
        "file_path": str(path),
        "sheet_names": res.sheet_names,
        "total_formulas": res.total_formulas,
        "has_uncomputed_formulas": res.has_uncomputed_formulas,
        "formula_cells": formula_summary,
        "semantic_translation_prompts": semantic_prompts,
    }


def create_mcp_server():
    """Create and configure the FastMCP server instance."""
    try:
        from fastmcp import FastMCP

        mcp = FastMCP("Docling Hybrid Document Ingestion Server")

        @mcp.tool()
        def convert_document(file_path: str, enable_formula_semantics: bool = True) -> str:
            """Convert any document (PDF, DOCX, XLSX, PPTX, TXT, CSV) to RAG-ready Markdown."""
            return convert_document_to_markdown(file_path, enable_formula_semantics)

        @mcp.tool()
        def pdf_triage(file_path: str) -> str:
            """Triage PDF document pages between fast vector parsing, RapidOCR, and VLM."""
            return json.dumps(triage_pdf_document(file_path), indent=2)

        @mcp.tool()
        def excel_inspection(file_path: str) -> str:
            """Extract Excel formulas and cached values, with fallback evaluation and LLM prompts."""
            return json.dumps(inspect_excel_spreadsheet(file_path), indent=2)

        return mcp
    except ImportError:
        _log.warning("FastMCP is not installed. Returning None for MCP server.")
        return None


if __name__ == "__main__":
    server = create_mcp_server()
    if server:
        server.run()
    else:
        print("FastMCP not installed.")

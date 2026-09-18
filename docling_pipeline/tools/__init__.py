"""AI Tool interfaces and FastMCP server for the Docling Hybrid Document Pipeline."""

from docling_pipeline.tools.mcp_tool import (
    convert_document_to_markdown,
    create_mcp_server,
    inspect_excel_spreadsheet,
    triage_pdf_document,
)

__all__ = [
    "convert_document_to_markdown",
    "create_mcp_server",
    "inspect_excel_spreadsheet",
    "triage_pdf_document",
]

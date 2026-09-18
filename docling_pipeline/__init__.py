"""Docling Hybrid Document Ingestion & Processing Pipeline package."""

from docling_pipeline.pipeline import (
    HybridDocumentPipeline,
    ProcessedDocumentResult,
)
from docling_pipeline.spreadsheet.excel_inspector import (
    CellInfo,
    ExcelInspectionResult,
    SheetInspection,
    inspect_excel,
)
from docling_pipeline.spreadsheet.formula_semantic import (
    FormulaContext,
    FormulaSemanticMapper,
    FormulaTranslationPrompt,
)
from docling_pipeline.triage.pdf_triage import (
    DocumentTriageReport,
    ImageArtifact,
    PageTriageInfo,
    TriageMode,
    triage_pdf,
)
from docling_pipeline.vlm.annotation_hook import (
    AsyncVLMWorkerPool,
    VLMClassificationLabel,
    VLMProcessResult,
    attach_vlm_result,
    export_hybrid_markdown,
)
from docling_pipeline.tools.mcp_tool import (
    convert_document_to_markdown,
    create_mcp_server,
    inspect_excel_spreadsheet,
    triage_pdf_document,
)

__version__ = "0.1.0"

__all__ = [
    "HybridDocumentPipeline",
    "ProcessedDocumentResult",
    "CellInfo",
    "ExcelInspectionResult",
    "SheetInspection",
    "inspect_excel",
    "FormulaContext",
    "FormulaSemanticMapper",
    "FormulaTranslationPrompt",
    "DocumentTriageReport",
    "ImageArtifact",
    "PageTriageInfo",
    "TriageMode",
    "triage_pdf",
    "AsyncVLMWorkerPool",
    "VLMClassificationLabel",
    "VLMProcessResult",
    "attach_vlm_result",
    "export_hybrid_markdown",
    "convert_document_to_markdown",
    "create_mcp_server",
    "inspect_excel_spreadsheet",
    "triage_pdf_document",
]

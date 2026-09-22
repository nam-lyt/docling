"""Unified Hybrid Document Processing Pipeline.

Orchestrates multi-format ingestion (DOCX, XLSX, PPTX, PDF, TXT/CSV/LOG),
single-pass spreadsheet inspection with formula semantics, page-level PDF triage,
safe VLM diagram-to-Mermaid/table annotation, and high-quality Markdown output for AI/RAG.
"""

from __future__ import annotations

import asyncio
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import charset_normalizer
from PIL import Image as PILImage
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DocItemLabel, DoclingDocument, TextItem
from pydantic import BaseModel, Field

from docling_pipeline.spreadsheet.excel_inspector import (
    ExcelInspectionResult,
    cells_to_markdown_table,
    inspect_excel,
    render_cells_to_markdown,
)
from docling_pipeline.spreadsheet.formula_semantic import FormulaSemanticMapper
from docling_pipeline.triage.pdf_triage import DocumentTriageReport, triage_pdf
from docling_pipeline.vlm.annotation_hook import (
    AsyncVLMWorkerPool,
    export_hybrid_markdown,
)

_log = logging.getLogger(__name__)


class ProcessedDocumentResult(BaseModel):
    """Unified output from processing any document through the pipeline."""

    file_path: str
    file_format: str
    markdown: str
    total_pages_or_sheets: int = 1
    processing_time_seconds: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    formula_contexts_count: int = 0
    triage_summary: dict[str, Any] | None = None
    extracted_images: dict[str, bytes] = Field(default_factory=dict)


class HybridDocumentPipeline:
    """Enterprise-grade hybrid ingestion and processing pipeline."""

    def __init__(
        self,
        vlm_pool: AsyncVLMWorkerPool | None = None,
        vlm_server_url: str | None = None,
        vlm_model: str | None = None,
        vlm_api_type: str | None = None,
        vlm_no_proxy: str | bool | None = None,
        vlm_concurrency: int | None = None,
        vlm_timeout: float | None = None,
        vlm_max_time: float | None = None,
        vlm_max_tokens: int | None = None,
        vlm_temperature: float | None = None,
        vlm_max_dim: int | None = None,
        min_pdf_vector_chars: int = 50,
        enable_formula_semantics: bool = True,
        enable_vlm: bool = True,
    ) -> None:
        self.vlm_pool = vlm_pool or AsyncVLMWorkerPool(
            server_url=vlm_server_url,
            model_name=vlm_model,
            api_type=vlm_api_type,
            no_proxy=vlm_no_proxy,
            max_concurrency=vlm_concurrency,
            timeout_seconds=vlm_timeout,
            max_time=vlm_max_time,
            max_tokens=vlm_max_tokens,
            temperature=vlm_temperature,
            max_image_dim=vlm_max_dim,
        )
        self.min_pdf_vector_chars = min_pdf_vector_chars
        self.enable_formula_semantics = enable_formula_semantics
        self.enable_vlm = enable_vlm
        self._doc_converter: DocumentConverter | None = None

    def _get_converter(self, ocr_needed: bool = False) -> DocumentConverter:
        """Instantiate or configure DocumentConverter with selective OCR and cropped picture generation."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ocr_needed
        pipeline_options.generate_picture_images = True
        pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)
        return DocumentConverter(format_options={InputFormat.PDF: pdf_format_option})

    def _enhance_pictures_with_vlm(self, doc: DoclingDocument) -> tuple[int, dict[str, Any]]:
        """Inspect all PictureItems, skip icons < 100x100, and enhance candidates with VLM."""
        if not self.enable_vlm:
            return 0, {}

        import asyncio
        from docling_core.types.doc import PictureItem
        from docling_pipeline.vlm.annotation_hook import attach_vlm_result

        vlm_items: list[tuple[Any, str, str]] = []
        for item, _ in doc.iterate_items():
            if isinstance(item, PictureItem):
                try:
                    pil_img = item.get_image(doc)
                    if pil_img is not None:
                        w, h = pil_img.size
                        # Heuristic: skip decorative icons/logos < 100x100 px
                        if w >= 100 and h >= 100:
                            vlm_items.append((pil_img, item.self_ref or "", "Diagram or Chart"))
                except Exception as e:
                    _log.debug(f"Could not extract image from PictureItem: {e}")

        if not vlm_items:
            return 0, {}

        # Process candidates concurrently with VLM Worker Pool
        vlm_results = asyncio.run(self.vlm_pool.process_batch(vlm_items))

        # Safely attach results to PictureItem metadata without modifying AST structure
        enhanced_count = 0
        for item, _ in doc.iterate_items():
            if isinstance(item, PictureItem) and item.self_ref in vlm_results:
                attach_vlm_result(doc, item, vlm_results[item.self_ref])
                enhanced_count += 1

        return enhanced_count, vlm_results

    def process(self, file_path_or_bytes: Path | BytesIO | str, filename: str | None = None) -> ProcessedDocumentResult:
        """Process an input document of any supported format into RAG-ready Markdown."""
        start_time = time.perf_counter()

        if isinstance(file_path_or_bytes, (str, Path)):
            path = Path(file_path_or_bytes)
            ext = path.suffix.lower()
            file_name = path.name
            path_or_buf: Path | BytesIO = path
        else:
            ext = (Path(filename).suffix.lower()) if filename else ".bin"
            file_name = filename or "document"
            path_or_buf = file_path_or_bytes

        # 1. Spreadsheet (XLSX / XLS)
        if ext in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
            return self._process_spreadsheet(path_or_buf, file_name, start_time)

        # 2. PDF with Page-level Triage
        elif ext == ".pdf":
            return self._process_pdf(path_or_buf, file_name, start_time)

        # 3. Plain Text / CSV / Log (charset-normalizer)
        elif ext in [".txt", ".csv", ".tsv", ".log", ".json", ".xml", ".yaml", ".yml"]:
            return self._process_stream_text(path_or_buf, file_name, ext, start_time)

        # 4. Word / PowerPoint / Other formats supported natively by Docling
        else:
            return self._process_native_docling(path_or_buf, file_name, ext, start_time)

    def _process_spreadsheet(
        self, source: Path | BytesIO, file_name: str, start_time: float
    ) -> ProcessedDocumentResult:
        """Process Excel spreadsheet via single-pass inspector, image extractor, and semantic formula mapper."""
        inspection = inspect_excel(source)

        sections: list[str] = [f"# Spreadsheet: {file_name}"]
        total_formula_contexts = 0
        extracted_images: dict[str, bytes] = {}

        stem = Path(file_name).stem

        for sheet in inspection.sheets:
            sections.append(f"\n## Sheet: {sheet.name}")

            # 1. Prepare embedded images, raw bytes, and candidate VLM batch
            sheet_img_map: dict[str, tuple[Any, PILImage.Image, str]] = {}
            vlm_batch: list[tuple[PILImage.Image, str, str]] = []

            for img_item in sheet.images:
                if img_item.raw_bytes:
                    img_filename = f"{stem}_{sheet.name}_{img_item.coordinate}.png"
                    rel_img_path = f"images/{img_filename}"
                    extracted_images[rel_img_path] = img_item.raw_bytes

                    try:
                        pil_img = PILImage.open(BytesIO(img_item.raw_bytes))
                        sheet_img_map[img_item.coordinate] = (img_item, pil_img, rel_img_path)
                        # Heuristic: skip icons/decorations < 100x100 px from VLM
                        if pil_img.width >= 100 and pil_img.height >= 100:
                            vlm_batch.append(
                                (
                                    pil_img,
                                    img_item.coordinate,
                                    f"Embedded visual in sheet '{sheet.name}' at cell {img_item.coordinate} ({img_item.width}x{img_item.height})",
                                )
                            )
                    except Exception as e:
                        _log.warning(
                            "Could not open embedded image at %s: %s",
                            img_item.coordinate,
                            e,
                        )

            vlm_results: dict[str, Any] = {}
            if self.enable_vlm and vlm_batch:
                try:
                    vlm_results = asyncio.run(
                        self.vlm_pool.process_batch(vlm_batch)
                    )
                except Exception as e:
                    _log.warning(
                        "VLM processing error for spreadsheet images: %s", e
                    )

            # 2. Render sheet content with chronological row-interleaving
            blocks = sheet.to_row_ordered_blocks()
            for block_type, block_data in blocks:
                if block_type == "cells":
                    rendered_sections = render_cells_to_markdown(block_data)
                    for sec in rendered_sections:
                        if sec.strip():
                            sections.append(sec)
                elif block_type == "image":
                    img_coord = block_data.coordinate
                    img_tuple = sheet_img_map.get(img_coord)
                    rel_img_path = (
                        img_tuple[2]
                        if img_tuple
                        else f"images/{stem}_{sheet.name}_{img_coord}.png"
                    )

                    vlm_res = vlm_results.get(img_coord)
                    if (
                        vlm_res
                        and (
                            vlm_res.markdown_table
                            or vlm_res.mermaid_code
                            or vlm_res.latex_equation
                            or vlm_res.raw_explanation
                        )
                        and vlm_res.confidence >= 0.8
                    ):
                        # True VLM extraction: replace image directly with structured content without redundant comments
                        sections.append(vlm_res.render_markdown())
                    else:
                        # Clean fallback: emit standard image link, NO FAKE TEXT!
                        sections.append(
                            f"![Image at cell {img_coord} ({block_data.width}x{block_data.height})]({rel_img_path})"
                        )

            # 3. Formula Logic & Semantic Context
            if self.enable_formula_semantics and sheet.formula_cells_count > 0:
                prompt_container = FormulaSemanticMapper.map_sheet_formulas(sheet)
                total_formula_contexts += len(prompt_container.contexts)
                if prompt_container.contexts:
                    sections.append("\n### 📐 Formula Logic & Semantic Context")
                    for ctx in prompt_container.contexts:
                        ref_info = ", ".join(
                            f"{k} ('{v}')"
                            for k, v in ctx.referenced_headers.items()
                        ) or ", ".join(ctx.referenced_cells)
                        sections.append(
                            f"- **`{ctx.coordinate}`** (`={ctx.formula}`): "
                            f"Evaluates to `{ctx.effective_value}` in column *'{ctx.column_header or ctx.col_letter}'* "
                            f"(references: {ref_info})"
                        )

        full_md = "\n\n".join(sections)
        elapsed = round(time.perf_counter() - start_time, 3)

        return ProcessedDocumentResult(
            file_path=str(source) if isinstance(source, Path) else file_name,
            file_format="xlsx",
            markdown=full_md,
            total_pages_or_sheets=len(inspection.sheets),
            processing_time_seconds=elapsed,
            formula_contexts_count=total_formula_contexts,
            extracted_images=extracted_images,
            metadata={
                "sheet_names": inspection.sheet_names,
                "total_formulas": inspection.total_formulas,
                "has_uncomputed_formulas": inspection.has_uncomputed_formulas,
                "total_images": sum(len(s.images) for s in inspection.sheets),
            },
        )

    def _process_pdf(
        self, source: Path | BytesIO, file_name: str, start_time: float
    ) -> ProcessedDocumentResult:
        """Process PDF with preliminary page-level triage and adaptive OCR."""
        # Run fast triage
        triage_report = triage_pdf(source, min_text_chars=self.min_pdf_vector_chars)

        # Determine if OCR is required
        ocr_needed = triage_report.requires_ocr
        converter = self._get_converter(ocr_needed=ocr_needed)

        # Convert document
        conv_res = converter.convert(source)
        doc = conv_res.document

        # Enhance pictures with VLM (converting charts -> tables, flowcharts -> Mermaid)
        vlm_enhanced_count, vlm_results = self._enhance_pictures_with_vlm(doc)

        # Render hybrid markdown (bung Mermaid and tables)
        full_md = export_hybrid_markdown(doc, vlm_results=vlm_results)
        elapsed = round(time.perf_counter() - start_time, 3)

        return ProcessedDocumentResult(
            file_path=str(source) if isinstance(source, Path) else file_name,
            file_format="pdf",
            markdown=full_md,
            total_pages_or_sheets=triage_report.total_pages,
            processing_time_seconds=elapsed,
            triage_summary=triage_report.get_summary(),
            metadata={
                "triage_decision": "OCR_ENABLED" if ocr_needed else "DIRECT_PARSE_FAST",
                "direct_pages": triage_report.direct_parse_pages,
                "ocr_pages": triage_report.ocr_pages,
                "hybrid_pages": triage_report.hybrid_pages,
                "vlm_enhanced_images_count": vlm_enhanced_count,
            },
        )

    def _process_stream_text(
        self, source: Path | BytesIO, file_name: str, ext: str, start_time: float
    ) -> ProcessedDocumentResult:
        """Decode and ingest plain text / stream files using charset-normalizer."""
        if isinstance(source, Path):
            raw_bytes = source.read_bytes()
        else:
            source.seek(0)
            raw_bytes = source.read()

        match = charset_normalizer.from_bytes(raw_bytes).best()
        decoded_text = str(match) if match else raw_bytes.decode("utf-8", errors="replace")
        encoding_name = match.encoding if match else "utf-8"

        # Format CSV / TSV into clean Markdown table if applicable
        full_md: str
        if ext == ".csv":
            lines = decoded_text.strip().splitlines()
            if lines:
                rows = [line.split(",") for line in lines]
                header = rows[0]
                body_rows = rows[1:]
                md_table = [
                    f"| {' | '.join(c.strip() for c in header)} |",
                    f"| {' | '.join(['---'] * len(header))} |",
                ]
                for r in body_rows:
                    md_table.append(f"| {' | '.join(c.strip() for c in r)} |")
                full_md = f"# Data Table: {file_name}\n\n" + "\n".join(md_table)
            else:
                full_md = f"*(Empty CSV: {file_name})*"
        else:
            full_md = f"# File: {file_name}\n\n```\n{decoded_text}\n```"

        elapsed = round(time.perf_counter() - start_time, 3)
        return ProcessedDocumentResult(
            file_path=str(source) if isinstance(source, Path) else file_name,
            file_format=ext.lstrip("."),
            markdown=full_md,
            total_pages_or_sheets=1,
            processing_time_seconds=elapsed,
            metadata={"detected_encoding": encoding_name},
        )

    def _process_native_docling(
        self, source: Path | BytesIO, file_name: str, ext: str, start_time: float
    ) -> ProcessedDocumentResult:
        """Process DOCX, PPTX, and HTML using native Docling backend (preserves oMath2Latex)."""
        converter = DocumentConverter()
        conv_res = converter.convert(source)
        doc = conv_res.document

        # Enhance raster pictures with VLM
        vlm_enhanced_count, vlm_results = self._enhance_pictures_with_vlm(doc)

        full_md = export_hybrid_markdown(doc, vlm_results=vlm_results)
        elapsed = round(time.perf_counter() - start_time, 3)

        return ProcessedDocumentResult(
            file_path=str(source) if isinstance(source, Path) else file_name,
            file_format=ext.lstrip("."),
            markdown=full_md,
            total_pages_or_sheets=1,
            processing_time_seconds=elapsed,
            metadata={
                "native_backend": "Docling DocumentConverter",
                "vlm_enhanced_images_count": vlm_enhanced_count,
            },
        )

"""Page-level PDF triage using pypdfium2 with heuristic image filtering.

Avoids the catastrophic failure mode where whole-file triage shuts off OCR
for scanned pages in hybrid documents (e.g. 9 digital pages + 1 signed scan).
Also filters out small decorative icons (< 100x100 px) from VLM queues.
"""

from __future__ import annotations

import enum
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)


class TriageMode(str, enum.Enum):
    """Execution triage decision per page."""

    DIRECT_PARSE = "direct_parse"  # Sufficient vector text, bypass OCR (0% OCR overhead)
    OCR_PAGE = "ocr_page"  # Scanned image / negligible text, requires RapidOCR
    HYBRID = "hybrid"  # Has vector text AND significant diagrams/charts (VLM candidate)


class ImageArtifact(BaseModel):
    """Metadata for an image found on a PDF page."""

    index: int
    px_width: int
    px_height: int
    pdf_bounds: tuple[float, float, float, float]
    is_decorative: bool = False
    is_candidate_for_vlm: bool = True


class PageTriageInfo(BaseModel):
    """Detailed triage assessment for a single PDF page."""

    page_number: int  # 1-based index
    char_count: int
    word_count: int
    text_snippet: str
    total_images_count: int = 0
    decorative_images_count: int = 0
    vlm_candidate_images_count: int = 0
    images: list[ImageArtifact] = Field(default_factory=list)
    triage_mode: TriageMode
    decision_reason: str


class DocumentTriageReport(BaseModel):
    """Aggregated triage report for the entire PDF document."""

    file_path: str
    total_pages: int
    direct_parse_pages: list[int] = Field(default_factory=list)
    ocr_pages: list[int] = Field(default_factory=list)
    hybrid_pages: list[int] = Field(default_factory=list)
    pages: list[PageTriageInfo] = Field(default_factory=list)

    @property
    def requires_ocr(self) -> bool:
        """True if at least one page requires OCR processing."""
        return len(self.ocr_pages) > 0

    @property
    def has_vlm_candidates(self) -> bool:
        """True if any page has images qualifying for VLM chart/flowchart reasoning."""
        return any(p.vlm_candidate_images_count > 0 for p in self.pages)

    def get_summary(self) -> dict[str, Any]:
        """Compact summary of document triage for logging or AI tool response."""
        return {
            "total_pages": self.total_pages,
            "direct_parse_pages": self.direct_parse_pages,
            "ocr_pages": self.ocr_pages,
            "hybrid_pages": self.hybrid_pages,
            "requires_ocr": self.requires_ocr,
            "has_vlm_candidates": self.has_vlm_candidates,
        }


def triage_pdf(
    pdf_path_or_bytes: Path | BytesIO | str,
    min_text_chars: int = 50,
    icon_threshold_px: int = 100,
) -> DocumentTriageReport:
    """Triage a PDF page by page to optimize OCR and VLM routing.

    Args:
        pdf_path_or_bytes: Path, string path, or BytesIO buffer containing the PDF.
        min_text_chars: Minimum character count to consider a page as native vector text.
        icon_threshold_px: Width/height below which images are classified as decorative icons.
    """
    file_path_str = (
        str(pdf_path_or_bytes)
        if isinstance(pdf_path_or_bytes, (Path, str))
        else "<stream>"
    )

    pdf: pdfium.PdfDocument
    if isinstance(pdf_path_or_bytes, (Path, str)):
        pdf = pdfium.PdfDocument(str(pdf_path_or_bytes))
    else:
        pdf_path_or_bytes.seek(0)
        pdf = pdfium.PdfDocument(pdf_path_or_bytes.read())

    total_pages = len(pdf)
    report = DocumentTriageReport(file_path=file_path_str, total_pages=total_pages)

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = pdf[page_idx]

        # 1. Extract vector text
        text_page = page.get_textpage()
        page_text = text_page.get_text_range()
        char_count = len(page_text.strip())
        word_count = len(page_text.strip().split())
        snippet = page_text.strip()[:100].replace("\n", " ")

        # 2. Inspect embedded images & filter decorative icons
        images: list[ImageArtifact] = []
        decorative_count = 0
        vlm_count = 0

        img_idx = 0
        for obj in page.get_objects():
            if obj.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE:
                try:
                    px_w, px_h = obj.get_px_size()
                    bounds = obj.get_bounds()
                    is_decorative = px_w < icon_threshold_px and px_h < icon_threshold_px
                    is_vlm_candidate = not is_decorative

                    if is_decorative:
                        decorative_count += 1
                    else:
                        vlm_count += 1

                    images.append(
                        ImageArtifact(
                            index=img_idx,
                            px_width=px_w,
                            px_height=px_h,
                            pdf_bounds=bounds,
                            is_decorative=is_decorative,
                            is_candidate_for_vlm=is_vlm_candidate,
                        )
                    )
                    img_idx += 1
                except Exception as e:
                    _log.debug(f"Could not read image object {img_idx} on page {page_num}: {e}")

        # 3. Determine triage mode
        if char_count < min_text_chars:
            mode = TriageMode.OCR_PAGE
            reason = f"Low vector text ({char_count} chars < {min_text_chars} threshold); scanned page requiring OCR"
            report.ocr_pages.append(page_num)
        elif vlm_count > 0:
            mode = TriageMode.HYBRID
            reason = f"Vector text present ({char_count} chars) with {vlm_count} significant diagram/chart images"
            report.hybrid_pages.append(page_num)
        else:
            mode = TriageMode.DIRECT_PARSE
            reason = f"Clean vector text ({char_count} chars); bypass OCR"
            report.direct_parse_pages.append(page_num)

        page_info = PageTriageInfo(
            page_number=page_num,
            char_count=char_count,
            word_count=word_count,
            text_snippet=snippet,
            total_images_count=len(images),
            decorative_images_count=decorative_count,
            vlm_candidate_images_count=vlm_count,
            images=images,
            triage_mode=mode,
            decision_reason=reason,
        )
        report.pages.append(page_info)

    return report

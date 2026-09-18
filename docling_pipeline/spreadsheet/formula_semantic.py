"""Semantic mapping and LLM prompt generation for Excel formulas.

This module maps raw cell coordinates and Excel formulas (e.g., =B5*C5) into
business-meaningful context (e.g., Total Revenue = Quantity * Unit Price)
to generate high-quality prompts and structured outputs for LLM reasoning and RAG.
"""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field

from docling_pipeline.spreadsheet.excel_inspector import (
    CellInfo,
    SheetInspection,
    split_coordinate,
)

_CELL_REF_PATTERN = re.compile(r"\b([A-Z]+[0-9]+)\b")


class FormulaContext(BaseModel):
    """Contextual metadata surrounding a formula cell."""

    sheet_name: str
    coordinate: str
    col_letter: str
    row: int
    formula: str
    effective_value: Any | None = None
    column_header: str | None = None
    row_label: str | None = None
    referenced_cells: list[str] = Field(default_factory=list)
    referenced_headers: dict[str, str] = Field(default_factory=dict)

    def generate_prompt(self, target_lang: str = "en") -> str:
        """Generate a concise, high-context prompt for an LLM to interpret this formula."""
        row_ctx = f" for '{self.row_label}'" if self.row_label else ""
        col_ctx = f"'{self.column_header}'" if self.column_header else f"column {self.col_letter}"

        ref_desc = []
        for ref_coord, header in self.referenced_headers.items():
            ref_desc.append(f"{ref_coord} ('{header}')")
        ref_text = ", ".join(ref_desc) if ref_desc else ", ".join(self.referenced_cells)

        if target_lang == "vi":
            return (
                f"Trong bảng '{self.sheet_name}', ô {self.coordinate} ({col_ctx}{row_ctx}) "
                f"chứa công thức: `={self.formula}` với giá trị tính ra là `{self.effective_value}`. "
                f"Các ô tham chiếu gồm: {ref_text}. "
                f"Hãy giải nghĩa ngắn gọn công thức này theo ngữ cảnh kinh doanh/nghiệp vụ."
            )
        return (
            f"In sheet '{self.sheet_name}', cell {self.coordinate} ({col_ctx}{row_ctx}) "
            f"has formula: `={self.formula}` with calculated value `{self.effective_value}`. "
            f"Referenced inputs: {ref_text}. "
            f"Explain the business logic and meaning of this formula in concise natural language."
        )


class FormulaTranslationPrompt(BaseModel):
    """Batch prompt container for all formulas in a workbook or sheet."""

    sheet_name: str
    contexts: list[FormulaContext] = Field(default_factory=list)

    def to_llm_prompt(self, target_lang: str = "en") -> str:
        """Render full LLM prompt for translating all formulas in the sheet."""
        if not self.contexts:
            return ""

        items_text = "\n".join(
            f"{i+1}. {ctx.generate_prompt(target_lang)}"
            for i, ctx in enumerate(self.contexts)
        )

        if target_lang == "vi":
            return (
                f"Bạn là chuyên gia phân tích dữ liệu tài chính & nghiệp vụ.\n"
                f"Dưới đây là các công thức trong trang tính '{self.sheet_name}'. "
                f"Hãy dịch từng công thức thành quy tắc nghiệp vụ dễ hiểu:\n\n"
                f"{items_text}\n\n"
                f"Trả về định dạng:\n"
                f"- [Tọa độ] Ý nghĩa nghiệp vụ: <Mô tả ngắn gọn>"
            )

        return (
            f"You are a business & financial data analyst.\n"
            f"Below are the formulas extracted from sheet '{self.sheet_name}'. "
            f"Translate each formula into clear business logic:\n\n"
            f"{items_text}\n\n"
            f"Format output as:\n"
            f"- [Coordinate] Business Meaning: <concise explanation>"
        )


class FormulaSemanticExplanation(BaseModel):
    """Structured LLM explanation for an individual formula cell."""

    coordinate: str
    business_rule: str
    natural_language_explanation: str
    latex_formula: str | None = None


class FormulaSemanticMapper:
    """Extracts business headers, row labels, and builds semantic prompts."""

    @staticmethod
    def _find_header_for_col(sheet: SheetInspection, col_letter: str, formula_row: int) -> str | None:
        """Find the column header above the formula cell (usually row 1 or nearest non-empty top cell)."""
        # Look from row 1 up to formula_row - 1
        for r in range(1, formula_row):
            coord = f"{col_letter}{r}"
            cell = sheet.get_cell(coord)
            if cell and cell.display_text().strip():
                return cell.display_text().strip()
        return None

    @staticmethod
    def _find_row_label_for_row(sheet: SheetInspection, row_idx: int, formula_col_letter: str) -> str | None:
        """Find the row label to the left of the formula cell (usually column A or B)."""
        formula_col_idx = sheet.get_cell(f"{formula_col_letter}{row_idx}")
        max_col = formula_col_idx.col if formula_col_idx else 10

        for col_letter in ["A", "B"]:
            coord = f"{col_letter}{row_idx}"
            cell = sheet.get_cell(coord)
            if cell and cell.col < max_col and cell.display_text().strip():
                return cell.display_text().strip()
        return None

    @classmethod
    def map_sheet_formulas(cls, sheet: SheetInspection) -> FormulaTranslationPrompt:
        """Extract all formula contexts from a sheet and build a structured prompt."""
        contexts: list[FormulaContext] = []

        for cell in sheet.cells:
            if not cell.is_formula or not cell.formula:
                continue

            col_letter, row_idx = split_coordinate(cell.coordinate)
            col_header = cls._find_header_for_col(sheet, col_letter, row_idx)
            row_label = cls._find_row_label_for_row(sheet, row_idx, col_letter)

            # Find referenced cell coordinates (e.g. A1, B2)
            ref_matches = _CELL_REF_PATTERN.findall(cell.formula.upper())
            unique_refs = sorted(set(ref_matches))

            referenced_headers: dict[str, str] = {}
            for ref in unique_refs:
                try:
                    ref_col, ref_row = split_coordinate(ref)
                    header = cls._find_header_for_col(sheet, ref_col, ref_row)
                    if header:
                        referenced_headers[ref] = header
                except ValueError:
                    pass

            context = FormulaContext(
                sheet_name=sheet.name,
                coordinate=cell.coordinate,
                col_letter=col_letter,
                row=row_idx,
                formula=cell.formula,
                effective_value=cell.effective_value,
                column_header=col_header,
                row_label=row_label,
                referenced_cells=unique_refs,
                referenced_headers=referenced_headers,
            )
            contexts.append(context)

        return FormulaTranslationPrompt(
            sheet_name=sheet.name,
            contexts=contexts,
        )

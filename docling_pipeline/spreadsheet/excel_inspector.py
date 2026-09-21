"""Single-pass Excel inspector extracting formulas (<f>) and cached values (<v>).

This module solves the common data_only=True limitation where programmatically
generated Excel spreadsheets (e.g., from ERP or pandas) return None for formula
cells that have never been opened and saved in Microsoft Excel.
"""

from __future__ import annotations

import logging
import posixpath
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Final
from zipfile import ZipFile

from lxml import etree
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

# Namespaces commonly found in OpenXML spreadsheet documents
_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NAMESPACES = {
    "s": _NS_MAIN,
    "r": _NS_REL,
}

# Safe XML parser preventing XXE and entity expansion
_SAFE_XML_PARSER: Final = etree.XMLParser(
    resolve_entities=False,
    load_dtd=False,
    no_network=True,
    dtd_validation=False,
)

_COORD_PATTERN = re.compile(r"^([A-Z]+)([0-9]+)$")


def col_letter_to_index(col_letter: str) -> int:
    """Convert column letters (e.g. 'A', 'Z', 'AA') to 1-based column index."""
    idx = 0
    for char in col_letter.upper():
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx


def col_index_to_letter(col_idx: int) -> str:
    """Convert 1-based column index to column letters."""
    chars: list[str] = []
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        chars.append(chr(65 + remainder))
    return "".join(reversed(chars))


def split_coordinate(coordinate: str) -> tuple[str, int]:
    """Split coordinate like 'B12' into ('B', 12)."""
    match = _COORD_PATTERN.match(coordinate.upper())
    if not match:
        raise ValueError(f"Invalid cell coordinate: {coordinate}")
    col_str, row_str = match.groups()
    return col_str, int(row_str)


class CellInfo(BaseModel):
    """Detailed metadata for an individual Excel cell."""

    coordinate: str
    col_letter: str
    row: int
    col: int
    cell_type: str = "n"
    formula: str | None = None
    cached_value: Any | None = None
    evaluated_value: Any | None = None
    effective_value: Any | None = None
    is_formula: bool = False
    needs_semantic_translation: bool = False

    def display_text(self) -> str:
        """Format value for user or LLM presentation."""
        if self.effective_value is not None:
            return str(self.effective_value)
        if self.formula is not None:
            return f"={self.formula}"
        return ""


class SheetImage(BaseModel):
    """Metadata for an image embedded in a worksheet."""

    image_id: str
    col: int
    row: int
    coordinate: str
    target_path: str
    width: int
    height: int
    raw_bytes: bytes = Field(default=b"", repr=False)


class SheetInspection(BaseModel):
    """Inspection output for a single Excel worksheet."""

    name: str
    dimension: str | None = None
    total_cells: int = 0
    formula_cells_count: int = 0
    missing_cache_formula_count: int = 0
    cells: list[CellInfo] = Field(default_factory=list)
    cells_by_coordinate: dict[str, CellInfo] = Field(default_factory=dict)
    images: list[SheetImage] = Field(default_factory=list)

    def get_cell(self, coordinate: str) -> CellInfo | None:
        """Get cell by coordinate (e.g. 'A1')."""
        return self.cells_by_coordinate.get(coordinate.upper())

    def to_matrix(self) -> list[list[str]]:
        """Convert sheet cells to a 2D matrix of display values."""
        if not self.cells:
            return []
        max_row = max(c.row for c in self.cells)
        max_col = max(c.col for c in self.cells)
        matrix = [["" for _ in range(max_col)] for _ in range(max_row)]
        for c in self.cells:
            matrix[c.row - 1][c.col - 1] = c.display_text()
        return matrix

    def to_markdown_table(self) -> str:
        """Convert sheet to standard markdown table, omitting fully empty rows."""
        matrix = self.to_matrix()
        if not matrix:
            return f"*(Empty sheet: {self.name})*"
        header = matrix[0]
        rows = matrix[1:] if len(matrix) > 1 else []
        md_lines = [
            f"| {' | '.join(header)} |",
            f"| {' | '.join(['---'] * len(header))} |",
        ]
        for row in rows:
            if any(cell.strip() for cell in row):
                md_lines.append(f"| {' | '.join(row)} |")
        return "\n".join(md_lines)

    def to_row_ordered_blocks(self) -> list[tuple[str, list[CellInfo] | SheetImage]]:
        """Group sheet contents into chronological row-ordered blocks (cells vs images).

        Returns a list of tuples:
        - ("cells", list_of_cells)
        - ("image", sheet_image)
        """
        if not self.images:
            return [("cells", self.cells)] if self.cells else []

        sorted_images = sorted(self.images, key=lambda im: (im.row, im.col))
        sorted_cells = sorted(self.cells, key=lambda c: (c.row, c.col))

        blocks: list[tuple[str, list[CellInfo] | SheetImage]] = []
        cell_idx = 0
        n_cells = len(sorted_cells)

        for img in sorted_images:
            cells_before: list[CellInfo] = []
            while cell_idx < n_cells:
                c = sorted_cells[cell_idx]
                if (c.row < img.row) or (c.row == img.row and c.col < img.col):
                    cells_before.append(c)
                    cell_idx += 1
                else:
                    break

            if cells_before:
                blocks.append(("cells", cells_before))

            blocks.append(("image", img))

        cells_after = sorted_cells[cell_idx:]
        if cells_after:
            blocks.append(("cells", cells_after))

        return blocks


def cells_to_markdown_table(cells: list[CellInfo]) -> str:
    """Format an arbitrary subset of cells into a clean Markdown table."""
    non_empty = [c for c in cells if c.display_text().strip()]
    if not non_empty:
        return ""

    active_cols = sorted({c.col for c in non_empty})
    distinct_rows = sorted({c.row for c in non_empty})
    cell_map = {(c.row, c.col): c.display_text() for c in non_empty}

    grid: list[list[str]] = []
    for r in distinct_rows:
        row_vals = [cell_map.get((r, c_idx), "") for c_idx in active_cols]
        grid.append(row_vals)

    if not grid:
        return ""

    # If only 1 row
    if len(grid) == 1:
        first_row = grid[0]
        if len(first_row) == 1:
            return first_row[0]
        return (
            f"| {' | '.join(first_row)} |\n"
            f"| {' | '.join(['---'] * len(first_row))} |"
        )

    header = grid[0]
    data_rows = grid[1:]
    md_lines = [
        f"| {' | '.join(header)} |",
        f"| {' | '.join(['---'] * len(header))} |",
    ]
    for r_vals in data_rows:
        if any(v.strip() for v in r_vals):
            md_lines.append(f"| {' | '.join(r_vals)} |")
    return "\n".join(md_lines)



class ExcelInspectionResult(BaseModel):
    """Aggregate result from inspecting an Excel workbook."""

    file_path: str
    sheet_names: list[str] = Field(default_factory=list)
    sheets: list[SheetInspection] = Field(default_factory=list)
    total_formulas: int = 0
    has_uncomputed_formulas: bool = False

    def get_sheet(self, name: str) -> SheetInspection | None:
        """Get sheet inspection by name."""
        for s in self.sheets:
            if s.name == name:
                return s
        return None

    def get_formula_summary(self) -> list[dict[str, Any]]:
        """Return a structured summary of all formula cells across sheets."""
        summary: list[dict[str, Any]] = []
        for sheet in self.sheets:
            for cell in sheet.cells:
                if cell.is_formula:
                    summary.append(
                        {
                            "sheet": sheet.name,
                            "coordinate": cell.coordinate,
                            "formula": cell.formula,
                            "cached_value": cell.cached_value,
                            "effective_value": cell.effective_value,
                            "needs_semantic_translation": cell.needs_semantic_translation,
                        }
                    )
        return summary


def _evaluate_basic_formula(
    formula: str,
    cells_dict: dict[str, CellInfo],
) -> Any | None:
    """Best-effort fallback evaluation for common basic Excel formulas.

    Used when cached_value is None because the workbook was generated programmatically.
    Supports SUM, AVERAGE, COUNT, MIN, MAX, and basic arithmetic between cell references.
    """
    clean_formula = formula.strip().lstrip("=").upper()

    # Match functions like SUM(A1:A5) or SUM(A1, B1, C1)
    func_match = re.match(r"^([A-Z]+)\(([^)]+)\)$", clean_formula)
    if func_match:
        func_name, args_str = func_match.groups()
        args = [arg.strip() for arg in args_str.split(",")]
        values: list[float] = []

        for arg in args:
            if ":" in arg:  # Range coordinate, e.g. A1:B3
                parts = arg.split(":")
                if len(parts) == 2:
                    start_col_str, start_row = split_coordinate(parts[0])
                    end_col_str, end_row = split_coordinate(parts[1])
                    start_col = col_letter_to_index(start_col_str)
                    end_col = col_letter_to_index(end_col_str)

                    for r in range(
                        min(start_row, end_row), max(start_row, end_row) + 1
                    ):
                        for c in range(
                            min(start_col, end_col), max(start_col, end_col) + 1
                        ):
                            coord = f"{col_index_to_letter(c)}{r}"
                            cell = cells_dict.get(coord)
                            if cell and cell.effective_value is not None:
                                try:
                                    values.append(float(cell.effective_value))
                                except (ValueError, TypeError):
                                    pass
            else:  # Single cell or constant
                cell = cells_dict.get(arg)
                if cell and cell.effective_value is not None:
                    try:
                        values.append(float(cell.effective_value))
                    except (ValueError, TypeError):
                        pass
                else:
                    try:
                        values.append(float(arg))
                    except ValueError:
                        pass

        if not values:
            return None

        if func_name == "SUM":
            return sum(values)
        if func_name == "AVERAGE":
            return sum(values) / len(values)
        if func_name == "COUNT":
            return len(values)
        if func_name == "MIN":
            return min(values)
        if func_name == "MAX":
            return max(values)

    # Match simple binary arithmetic like A1+B1, A1*B1, A1-B1, A1/B1
    arith_match = re.match(
        r"^([A-Z]+[0-9]+)\s*([\+\-\*\/])\s*([A-Z]+[0-9]+)$", clean_formula
    )
    if arith_match:
        c1, op, c2 = arith_match.groups()
        val1 = cells_dict.get(c1)
        val2 = cells_dict.get(c2)
        if (
            val1
            and val1.effective_value is not None
            and val2
            and val2.effective_value is not None
        ):
            try:
                n1 = float(val1.effective_value)
                n2 = float(val2.effective_value)
                if op == "+":
                    return n1 + n2
                if op == "-":
                    return n1 - n2
                if op == "*":
                    return n1 * n2
                if op == "/":
                    return n1 / n2 if n2 != 0 else None
            except (ValueError, TypeError, ZeroDivisionError):
                return None

    return None


def _parse_shared_strings(zip_file: ZipFile) -> list[str]:
    """Parse the shared strings table (xl/sharedStrings.xml)."""
    shared_strings: list[str] = []
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return shared_strings

    try:
        content = zip_file.read("xl/sharedStrings.xml")
        root = etree.fromstring(content, parser=_SAFE_XML_PARSER)
        # Each <si> (string item) contains <t> or multiple <r><t> runs
        for si in root.findall(f"{{{_NS_MAIN}}}si"):
            texts = [
                t.text
                for t in si.iter(f"{{{_NS_MAIN}}}t")
                if t.text is not None
            ]
            shared_strings.append("".join(texts))
    except Exception as e:
        _log.warning(f"Could not parse shared strings: {e}")

    return shared_strings


def _parse_workbook_sheets(zip_file: ZipFile) -> list[tuple[str, str]]:
    """Parse xl/workbook.xml and relationships to map sheet names to xml file paths."""
    sheets: list[tuple[str, str]] = []
    if (
        "xl/workbook.xml" not in zip_file.namelist()
        or "xl/_rels/workbook.xml.rels" not in zip_file.namelist()
    ):
        return sheets

    try:
        rels_xml = zip_file.read("xl/_rels/workbook.xml.rels")
        rels_root = etree.fromstring(rels_xml, parser=_SAFE_XML_PARSER)
        r_id_to_target: dict[str, str] = {}
        for rel in rels_root.findall(
            "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"
        ):
            r_id = rel.get("Id")
            target = rel.get("Target")
            if r_id and target:
                clean_target = target.lstrip("/")
                if not clean_target.startswith("xl/"):
                    clean_target = f"xl/{clean_target}"
                r_id_to_target[r_id] = clean_target

        wb_xml = zip_file.read("xl/workbook.xml")
        wb_root = etree.fromstring(wb_xml, parser=_SAFE_XML_PARSER)
        for sheet_elem in wb_root.iter(f"{{{_NS_MAIN}}}sheet"):
            name = sheet_elem.get("name")
            r_id = sheet_elem.get(f"{{{_NS_REL}}}id")
            if name and r_id and r_id in r_id_to_target:
                sheets.append((name, r_id_to_target[r_id]))
    except Exception as e:
        _log.warning(f"Could not parse workbook sheet relationships: {e}")

    return sheets


def _parse_sheet_drawings(zip_file: ZipFile, sheet_path: str) -> list[SheetImage]:
    """Parse drawings and embedded images anchored in a worksheet."""
    images: list[SheetImage] = []
    sheet_dir = posixpath.dirname(sheet_path)
    sheet_file = posixpath.basename(sheet_path)
    sheet_rels_path = posixpath.join(sheet_dir, "_rels", f"{sheet_file}.rels")

    if sheet_rels_path not in zip_file.namelist():
        return images

    try:
        rels_xml = zip_file.read(sheet_rels_path)
        rels_root = etree.fromstring(rels_xml, parser=_SAFE_XML_PARSER)

        drawing_targets: list[str] = []
        for rel in rels_root.iter():
            rel_type = rel.get("Type", "")
            if rel_type.endswith("/drawing"):
                target = rel.get("Target")
                if target:
                    resolved_target = posixpath.normpath(
                        posixpath.join(sheet_dir, target)
                    ).lstrip("/")
                    drawing_targets.append(resolved_target)

        for drawing_path in drawing_targets:
            if drawing_path not in zip_file.namelist():
                continue

            drawing_dir = posixpath.dirname(drawing_path)
            drawing_file = posixpath.basename(drawing_path)
            drawing_rels_path = posixpath.join(
                drawing_dir, "_rels", f"{drawing_file}.rels"
            )

            # Map relationship ID to image target path
            r_id_to_image: dict[str, str] = {}
            if drawing_rels_path in zip_file.namelist():
                d_rels_xml = zip_file.read(drawing_rels_path)
                d_rels_root = etree.fromstring(d_rels_xml, parser=_SAFE_XML_PARSER)
                for d_rel in d_rels_root.iter():
                    if d_rel.get("Type", "").endswith("/image"):
                        r_id = d_rel.get("Id")
                        target = d_rel.get("Target")
                        if r_id and target:
                            r_id_to_image[r_id] = posixpath.normpath(
                                posixpath.join(drawing_dir, target)
                            ).lstrip("/")

            # Parse drawing XML anchors
            drawing_xml = zip_file.read(drawing_path)
            drawing_root = etree.fromstring(drawing_xml, parser=_SAFE_XML_PARSER)

            for anchor in drawing_root:
                tag = anchor.tag
                if not (tag.endswith("Anchor") or "Anchor" in tag):
                    continue

                col_1based = 1
                row_1based = 1
                for elem in anchor.iter():
                    if elem.tag.endswith("}from") or elem.tag == "from":
                        for c in elem.iter():
                            if (
                                (c.tag.endswith("}col") or c.tag == "col")
                                and c.text
                                and c.text.strip().isdigit()
                            ):
                                col_1based = int(c.text.strip()) + 1
                            elif (
                                (c.tag.endswith("}row") or c.tag == "row")
                                and c.text
                                and c.text.strip().isdigit()
                            ):
                                row_1based = int(c.text.strip()) + 1
                        break

                coord = f"{col_index_to_letter(col_1based)}{row_1based}"

                r_id = None
                for elem in anchor.iter():
                    if elem.tag.endswith("}blip") or elem.tag == "blip":
                        for attr_k, attr_v in elem.attrib.items():
                            if attr_k.endswith("embed") or attr_k == "embed":
                                r_id = attr_v
                                break
                        if r_id:
                            break

                if not r_id or r_id not in r_id_to_image:
                    continue

                img_path = r_id_to_image[r_id]
                if img_path not in zip_file.namelist():
                    continue

                raw_bytes = zip_file.read(img_path)
                w, h = 0, 0
                try:
                    from PIL import Image as PILImage

                    with PILImage.open(BytesIO(raw_bytes)) as pil_img:
                        w, h = pil_img.size
                except Exception as e:
                    _log.debug("Could not determine image dimensions: %s", e)

                images.append(
                    SheetImage(
                        image_id=r_id,
                        col=col_1based,
                        row=row_1based,
                        coordinate=coord,
                        target_path=img_path,
                        width=w,
                        height=h,
                        raw_bytes=raw_bytes,
                    )
                )

    except Exception as e:
        _log.warning(f"Could not parse sheet drawings for {sheet_path}: {e}")

    return images


def inspect_excel(path_or_stream: Path | BytesIO | str) -> ExcelInspectionResult:
    """Inspect Excel spreadsheet in a single pass.

    Extracts both formula (<f>) and cached value (<v>) for every cell,
    with fallback formula evaluation if cached value is None.
    """
    file_path_str = (
        str(path_or_stream)
        if isinstance(path_or_stream, (Path, str))
        else "<stream>"
    )
    result = ExcelInspectionResult(file_path=file_path_str)

    if isinstance(path_or_stream, (str, Path)):
        source_zip: Path | BytesIO = Path(path_or_stream)
    else:
        source_zip = path_or_stream
        source_zip.seek(0)

    with ZipFile(source_zip, "r") as z:
        # Check for zip-slip / traversal security
        for member in z.namelist():
            if member.startswith("/") or ".." in member:
                raise ValueError(
                    f"Insecure ZIP path detected in workbook: {member}"
                )

        shared_strings = _parse_shared_strings(z)
        sheet_info_list = _parse_workbook_sheets(z)

        # Fallback if relationships failed: look for sheet*.xml directly
        if not sheet_info_list:
            sheet_files = [
                name
                for name in z.namelist()
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            ]
            sheet_files.sort()
            sheet_info_list = [
                (f"Sheet{i}", file_path)
                for i, file_path in enumerate(sheet_files, 1)
            ]

        for sheet_name, sheet_path in sheet_info_list:
            if sheet_path not in z.namelist():
                continue

            result.sheet_names.append(sheet_name)
            sheet_inspection = SheetInspection(name=sheet_name)
            sheet_inspection.images = _parse_sheet_drawings(z, sheet_path)

            sheet_xml = z.read(sheet_path)
            root = etree.fromstring(sheet_xml, parser=_SAFE_XML_PARSER)

            # Extract dimension
            dim_elem = root.find(f"{{{_NS_MAIN}}}dimension")
            if dim_elem is not None and dim_elem.get("ref"):
                sheet_inspection.dimension = dim_elem.get("ref")

            # Parse cells in <sheetData>
            for c_elem in root.iter(f"{{{_NS_MAIN}}}c"):
                coord = c_elem.get("r")
                if not coord:
                    continue

                cell_type = c_elem.get("t", "n")
                col_letter, row_idx = split_coordinate(coord)
                col_idx = col_letter_to_index(col_letter)

                # Extract formula if present
                f_elem = c_elem.find(f"{{{_NS_MAIN}}}f")
                formula_text = f_elem.text if f_elem is not None else None
                is_formula = formula_text is not None

                # Extract cached value if present
                v_elem = c_elem.find(f"{{{_NS_MAIN}}}v")
                cached_val_raw = v_elem.text if v_elem is not None else None
                cached_value: Any | None = None

                if cell_type == "inlineStr":
                    is_elem = c_elem.find(f"{{{_NS_MAIN}}}is")
                    if is_elem is not None:
                        cached_value = "".join(
                            t.text
                            for t in is_elem.iter(f"{{{_NS_MAIN}}}t")
                            if t.text is not None
                        )
                elif cached_val_raw is not None:
                    if cell_type == "s":  # Shared string
                        try:
                            s_idx = int(cached_val_raw)
                            cached_value = (
                                shared_strings[s_idx]
                                if 0 <= s_idx < len(shared_strings)
                                else cached_val_raw
                            )
                        except ValueError:
                            cached_value = cached_val_raw
                    elif cell_type == "b":  # Boolean
                        cached_value = cached_val_raw == "1"
                    elif cell_type == "n":  # Number
                        try:
                            cached_value = (
                                int(cached_val_raw)
                                if cached_val_raw.isdigit()
                                else float(cached_val_raw)
                            )
                        except ValueError:
                            cached_value = cached_val_raw
                    else:
                        cached_value = cached_val_raw

                # Fallback evaluation if formula exists but cached_value is None
                evaluated_val: Any | None = None
                if is_formula and cached_value is None:
                    sheet_inspection.missing_cache_formula_count += 1
                    result.has_uncomputed_formulas = True
                    evaluated_val = _evaluate_basic_formula(
                        formula_text, sheet_inspection.cells_by_coordinate
                    )

                effective_value = (
                    cached_value if cached_value is not None else evaluated_val
                )
                needs_semantic = is_formula and (
                    len(formula_text) > 4 or any(fn in formula_text.upper() for fn in ["VLOOKUP", "XLOOKUP", "INDEX", "MATCH", "IF", "PMT", "NPV", "IRR"])
                )

                cell_info = CellInfo(
                    coordinate=coord,
                    col_letter=col_letter,
                    row=row_idx,
                    col=col_idx,
                    cell_type=cell_type,
                    formula=formula_text,
                    cached_value=cached_value,
                    evaluated_value=evaluated_val,
                    effective_value=effective_value,
                    is_formula=is_formula,
                    needs_semantic_translation=needs_semantic,
                )

                sheet_inspection.cells.append(cell_info)
                sheet_inspection.cells_by_coordinate[coord] = cell_info
                sheet_inspection.total_cells += 1
                if is_formula:
                    sheet_inspection.formula_cells_count += 1
                    result.total_formulas += 1

            result.sheets.append(sheet_inspection)

    return result

"""Safe VLM annotation hooks and hybrid markdown serialization.

Ensures that replacing images with Mermaid flowcharts or Markdown tables NEVER
breaks the DoclingDocument AST tree or causes downstream serialization crashes.
Also provides an async worker pool with concurrency limit, bounding box cropping,
and robust timeout/fallback handling.
"""

from __future__ import annotations

import asyncio
import base64
import enum
import io
import json
import logging
import os
import re
from typing import Any, Callable, Coroutine
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from PIL import Image as PILImage
from pydantic import BaseModel, Field

from docling_core.types.doc import (
    DescriptionAnnotation,
    DescriptionMetaField,
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    PictureMeta,
    TableItem,
    TextItem,
)

_log = logging.getLogger(__name__)


class VLMClassificationLabel(str, enum.Enum):
    """Semantic classification of an embedded image."""

    CHART = "chart"  # Bar chart, pie chart, line graph -> convert to Markdown table + insight
    FLOWCHART_DIAGRAM = "flowchart_diagram"  # Architecture, process flow -> convert to Mermaid
    TABLE_IMAGE = "table_image"  # Rasterized spreadsheet/table -> convert to Markdown table
    EQUATION_IMAGE = "equation_image"  # Raster math equation -> convert to LaTeX $...$
    OTHER = "other"  # Photo, illustration, logo -> describe in natural language


class VLMProcessResult(BaseModel):
    """Structured output from VLM reasoning over an image element."""

    element_id: str
    classification: VLMClassificationLabel
    mermaid_code: str | None = None
    markdown_table: str | None = None
    latex_equation: str | None = None
    business_insight: str | None = None
    raw_explanation: str
    confidence: float = 1.0

    def render_markdown(self) -> str:
        """Render the optimal markdown representation for this image."""
        blocks: list[str] = []

        if self.classification == VLMClassificationLabel.FLOWCHART_DIAGRAM and self.mermaid_code:
            clean_mermaid = self.mermaid_code.strip()
            if not clean_mermaid.startswith("```"):
                clean_mermaid = f"```mermaid\n{clean_mermaid}\n```"
            blocks.append(clean_mermaid)
        elif self.classification in (VLMClassificationLabel.CHART, VLMClassificationLabel.TABLE_IMAGE) and self.markdown_table:
            blocks.append(self.markdown_table.strip())
        elif self.classification == VLMClassificationLabel.EQUATION_IMAGE and self.latex_equation:
            clean_eq = self.latex_equation.strip().strip("$")
            blocks.append(f"$${clean_eq}$$")
        else:
            blocks.append(f"> **[Image Analysis]**: {self.raw_explanation}")

        if self.business_insight:
            blocks.append(f"> 💡 **Key Insight**: {self.business_insight.strip()}")

        return "\n\n".join(blocks)


def attach_vlm_result(
    doc: DoclingDocument,
    picture: PictureItem,
    result: VLMProcessResult,
) -> None:
    """Safely attach VLM result to PictureItem metadata without modifying AST structure.

    This preserves all document hierarchy, references, and bounding box provenance.
    """
    rendered_text = result.render_markdown()

    # 1. Update picture.meta.description (modern schema)
    if picture.meta is None:
        picture.meta = PictureMeta()
    picture.meta.description = DescriptionMetaField(
        confidence=result.confidence,
        created_by="vlm_hybrid_pipeline",
        text=rendered_text,
    )
    if picture.meta and picture.meta.description:                                                                                                                                                                                                    
        print(picture.meta.description.created_by)  # Prints: "vlm_hybrid_pipeline"                                                                                                                                                                  
        print(picture.meta.description.confidence) 

    # 2. Add DescriptionAnnotation for backward compatibility
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            picture.annotations.append(
                DescriptionAnnotation(
                    provenance="vlm_hybrid_pipeline",
                    text=rendered_text,
                )
            )
    except Exception as e:
        _log.debug(f"Could not append DescriptionAnnotation: {e}")


def export_hybrid_markdown(
    doc: DoclingDocument,
    vlm_results: dict[str, VLMProcessResult] | None = None,
) -> str:
    """Export DoclingDocument to Markdown, substituting VLM annotations for images.

    If a PictureItem has a VLM annotation (e.g. Mermaid flowchart or Table),
    it is emitted directly in-place rather than a generic image placeholder.
    """
    vlm_map = vlm_results or {}
    rendered_parts: list[str] = []

    # Iterate over document body items
    for item, _level in doc.iterate_items():
        if isinstance(item, PictureItem):
            element_id = item.self_ref or ""
            # Check explicit registry first, then item.meta.description
            vlm_res = vlm_map.get(element_id)
            if vlm_res is not None:
                rendered_parts.append(vlm_res.render_markdown())
            elif item.meta and item.meta.description and item.meta.description.text:
                rendered_parts.append(item.meta.description.text)
            else:
                # Default fallback: picture caption or standard marker
                caption_text = ""
                if item.captions:
                    caption_text = " - " + " ".join(c.text for c in item.captions if hasattr(c, "text"))
                rendered_parts.append(f"<!-- image{caption_text} -->")

        elif isinstance(item, TableItem):
            # Render table directly via markdown or html
            try:
                rendered_parts.append(item.export_to_markdown())
            except Exception:
                rendered_parts.append(item.text if hasattr(item, "text") else "")

        elif isinstance(item, TextItem):
            prefix = ""
            if item.label == DocItemLabel.TITLE:
                prefix = "# "
            elif item.label == DocItemLabel.SECTION_HEADER:
                prefix = "## "
            rendered_parts.append(f"{prefix}{item.text}")

        else:
            if hasattr(item, "text") and item.text:
                rendered_parts.append(item.text)

    return "\n\n".join(part for part in rendered_parts if part.strip())


def _normalize_chat_endpoint(url: str) -> str:
    """Normalize a server URL to an OpenAI-compatible /v1/chat/completions endpoint."""
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    parsed = urlparse(cleaned)
    if parsed.path in ("", "/"):
        return f"{cleaned}/v1/chat/completions"
    return cleaned


class AsyncVLMWorkerPool:
    """Async worker pool for local VLM / vLLM requests with bounded concurrency and timeout.

    Connects to a local or self-hosted VLM instance (e.g. vLLM server, Ollama, LM Studio,
    or any OpenAI-compatible Vision endpoint) with robust fallback handling.
    Prevents pipeline starvation when documents contain dozens of images.
    """

    def __init__(
        self,
        server_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        max_concurrency: int | None = None,
        timeout_seconds: float | None = None,
        max_image_dim: int | None = None,
        vlm_caller: Callable[[PILImage.Image, str], Coroutine[Any, Any, VLMProcessResult]] | None = None,
    ) -> None:
        # Concurrency limit
        if max_concurrency is not None:
            self.max_concurrency = max_concurrency
        else:
            try:
                self.max_concurrency = int(
                    os.environ.get("VLM_MAX_CONCURRENCY")
                    or os.environ.get("VLM_CONCURRENCY")
                    or "1"
                )
            except ValueError:
                self.max_concurrency = 1
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

        # Timeout limit
        if timeout_seconds is not None:
            self.timeout = timeout_seconds
        else:
            try:
                self.timeout = float(os.environ.get("VLM_TIMEOUT", "60.0"))
            except ValueError:
                self.timeout = 60.0

        # Max image dimension for vision context protection
        if max_image_dim is not None:
            self.max_image_dim = max_image_dim
        else:
            try:
                self.max_image_dim = int(os.environ.get("VLM_MAX_DIM", "1024"))
            except ValueError:
                self.max_image_dim = 1024

        # Configure local VLM / vLLM server parameters
        if server_url is not None:
            raw_url = server_url
        else:
            raw_url = (
                os.environ.get("VLLM_SERVER_URL")
                or os.environ.get("VLM_SERVER_URL")
                or os.environ.get("LOCAL_VLM_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or ""
            )
        self.server_url = _normalize_chat_endpoint(raw_url) if raw_url else ""
        self.model_name = (
            model_name
            or os.environ.get("VLLM_MODEL")
            or os.environ.get("VLM_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "Qwen/Qwen2-VL-7B-Instruct"
        )
        self.api_key = (
            api_key
            or os.environ.get("VLLM_API_KEY")
            or os.environ.get("VLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"
        )
        self.vlm_caller = vlm_caller or self._default_local_vlm_caller
        # Backward-compatibility alias
        self._default_mock_caller = self._default_local_vlm_caller

    async def _default_local_vlm_caller(self, image: PILImage.Image, element_id: str) -> VLMProcessResult:
        """Call local or self-hosted vLLM / OpenAI-compatible VLM server.

        Falls back to clean heuristic classification if the server is offline or unreachable.
        """
        if not self.server_url:
            return self._heuristic_fallback(image, element_id)

        import httpx

        # Downscale large image to prevent exceeding token context limit in vision models
        max_dim = self.max_image_dim
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            proc_img = image.resize((int(w * scale), int(h * scale)), PILImage.Resampling.LANCZOS)
        else:
            proc_img = image

        # Convert image to base64 JPEG
        buffered = io.BytesIO()
        rgb_image = proc_img.convert("RGB")
        rgb_image.save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt_text = (
            "Transcribe this document image faithfully.\n"
            "- If it contains a table, spreadsheet, or report: convert all rows and columns into a clean Markdown table.\n"
            "- If it is a flowchart or process diagram: generate Mermaid syntax starting with ```mermaid and ending with ```.\n"
            "- If it contains equations: convert to LaTeX $...$.\n"
            "- If it is text or a document: extract all text cleanly.\n"
            "Do not add speculative commentary or invented insights."
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.server_url, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        raw_content = choices[0].get("message", {}).get("content", "")
                        return self._parse_vlm_output(raw_content, element_id)
                else:
                    _log.warning(
                        f"Local VLM server returned HTTP {resp.status_code}: {resp.text[:200]}"
                    )
        except Exception as e:
            _log.debug(
                f"Local VLM server request to {self.server_url} failed ({e}), using heuristic fallback"
            )

        return self._heuristic_fallback(image, element_id)

    def _parse_vlm_output(self, raw_content: str, element_id: str) -> VLMProcessResult:
        """Parse structured fields from local VLM response (JSON or Markdown-embedded)."""
        text = raw_content.strip()
        parsed: dict[str, Any] = {}

        # 1. Try extracting JSON if wrapped in markdown code fence: ```json ... ``` or ``` ... ```
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
            except Exception:
                pass

        # 2. Try parsing entire text as direct JSON
        if not parsed:
            try:
                parsed = json.loads(text)
            except Exception:
                # Try finding first { and last }
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(text[start : end + 1])
                    except Exception:
                        pass

        if parsed and isinstance(parsed, dict):
            cls_str = str(parsed.get("classification", "other")).lower()
            label = VLMClassificationLabel.OTHER
            if "flowchart" in cls_str or "diagram" in cls_str:
                label = VLMClassificationLabel.FLOWCHART_DIAGRAM
            elif "chart" in cls_str:
                label = VLMClassificationLabel.CHART
            elif "table" in cls_str:
                label = VLMClassificationLabel.TABLE_IMAGE
            elif "equation" in cls_str or "math" in cls_str:
                label = VLMClassificationLabel.EQUATION_IMAGE

            return VLMProcessResult(
                element_id=element_id,
                classification=label,
                mermaid_code=parsed.get("mermaid_code"),
                markdown_table=parsed.get("markdown_table"),
                latex_equation=parsed.get("latex_equation"),
                business_insight=parsed.get("business_insight"),
                raw_explanation=parsed.get("description") or parsed.get("raw_explanation") or "Local VLM analyzed image",
                confidence=0.95,
            )

        # 3. If not strict JSON, extract markdown structures directly
        mermaid_match = re.search(r"```mermaid\s*(.*?)\s*```", text, re.DOTALL)
        if mermaid_match:
            return VLMProcessResult(
                element_id=element_id,
                classification=VLMClassificationLabel.FLOWCHART_DIAGRAM,
                mermaid_code=mermaid_match.group(1).strip(),
                raw_explanation="Flowchart diagram generated by local VLM",
                confidence=0.90,
            )

        # Check for markdown table
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
        if len(table_lines) >= 2:
            return VLMProcessResult(
                element_id=element_id,
                classification=VLMClassificationLabel.TABLE_IMAGE,
                markdown_table="\n".join(table_lines),
                raw_explanation="Table extracted by local VLM",
                confidence=0.90,
            )

        return VLMProcessResult(
            element_id=element_id,
            classification=VLMClassificationLabel.OTHER,
            raw_explanation=text or "Local VLM analyzed image",
            confidence=0.85,
        )

    def _heuristic_fallback(self, image: PILImage.Image, element_id: str) -> VLMProcessResult:
        """Clean heuristic fallback when server is offline - does NOT invent fake tables or insights."""
        w, h = image.size
        if w > h * 1.5:  # Wide -> typical flowchart / architecture diagram
            return VLMProcessResult(
                element_id=element_id,
                classification=VLMClassificationLabel.FLOWCHART_DIAGRAM,
                mermaid_code=None,
                business_insight=None,
                raw_explanation="",
                confidence=0.5,
            )
        else:  # Square/vertical -> typical chart or table
            return VLMProcessResult(
                element_id=element_id,
                classification=VLMClassificationLabel.CHART,
                markdown_table=None,
                business_insight=None,
                raw_explanation="",
                confidence=0.5,
            )

    async def process_image_with_fallback(
        self,
        image: PILImage.Image,
        element_id: str,
        fallback_description: str = "Image element",
    ) -> VLMProcessResult:
        """Process a single image within semaphore and timeout guards."""
        async with self.semaphore:
            try:
                result = await asyncio.wait_for(
                    self.vlm_caller(image, element_id),
                    timeout=self.timeout,
                )
                if result is not None:
                    return result
            except asyncio.TimeoutError:
                _log.warning(f"VLM processing timed out for {element_id} after {self.timeout}s; using fallback")
                return VLMProcessResult(
                    element_id=element_id,
                    classification=VLMClassificationLabel.OTHER,
                    raw_explanation=f"{fallback_description} (VLM timed out)",
                    confidence=0.5,
                )
            except Exception as e:
                _log.warning(f"VLM processing error for {element_id}: {e}; using fallback")

            return self._heuristic_fallback(image, element_id)

    async def process_batch(
        self,
        items: list[tuple[PILImage.Image, str, str]],
    ) -> dict[str, VLMProcessResult]:
        """Process a batch of images concurrently.

        Args:
            items: list of (image, element_id, fallback_description) tuples.
        """
        tasks = [
            self.process_image_with_fallback(img, el_id, fallback)
            for img, el_id, fallback in items
        ]
        results = await asyncio.gather(*tasks)
        return {r.element_id: r for r in results if r is not None}

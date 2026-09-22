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
            blocks.append(self.raw_explanation)

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
    """Async worker pool for local VLM / OpenAI-compatible requests with bounded concurrency and timeout.

    Matches test_llm.py calling style directly with trust_env=False (no proxy) and configurable max-time.
    """

    def __init__(
        self,
        server_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        max_concurrency: int | None = None,
        timeout_seconds: float | None = None,
        max_time: float | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_image_dim: int | None = None,
        vlm_caller: Callable[[PILImage.Image, str], Coroutine[Any, Any, VLMProcessResult]] | None = None,
        **kwargs: Any,
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

        # Timeout limit (--max-time, default 600.0s)
        effective_timeout = max_time if max_time is not None else timeout_seconds
        if effective_timeout is not None:
            self.timeout = float(effective_timeout)
        else:
            try:
                self.timeout = float(
                    os.environ.get("VLM_MAX_TIME")
                    or os.environ.get("VLM_TIMEOUT")
                    or os.environ.get("MAX_TIME")
                    or "600.0"
                )
            except ValueError:
                self.timeout = 600.0

        # Max tokens and temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        else:
            try:
                self.max_tokens = int(
                    os.environ.get("VLM_MAX_TOKENS")
                    or os.environ.get("MAX_TOKENS")
                    or "4096"
                )
            except ValueError:
                self.max_tokens = 4096

        if temperature is not None:
            self.temperature = temperature
        else:
            try:
                self.temperature = float(
                    os.environ.get("VLM_TEMPERATURE")
                    or os.environ.get("TEMPERATURE")
                    or "0.0"
                )
            except ValueError:
                self.temperature = 0.0

        # Max image dimension for vision context protection
        if max_image_dim is not None:
            self.max_image_dim = max_image_dim
        else:
            try:
                self.max_image_dim = int(os.environ.get("VLM_MAX_DIM", "1024"))
            except ValueError:
                self.max_image_dim = 1024

        # Configure server parameters
        if server_url is not None:
            raw_url = server_url
        else:
            raw_url = (
                os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("VLLM_SERVER_URL")
                or os.environ.get("VLM_SERVER_URL")
                or os.environ.get("LOCAL_VLM_URL")
                or ""
            )

        self.server_url = _normalize_chat_endpoint(raw_url) if raw_url else ""
        self.model_name = (
            model_name
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("VLLM_MODEL")
            or os.environ.get("VLM_MODEL")
            or "gemma-4-26B"
        )
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("VLLM_API_KEY")
            or os.environ.get("VLM_API_KEY")
            or "EMPTY"
        )
        self.vlm_caller = vlm_caller or self._default_local_vlm_caller
        self._default_mock_caller = self._default_local_vlm_caller

    async def _default_local_vlm_caller(self, image: PILImage.Image, element_id: str) -> VLMProcessResult:
        """Call local or self-hosted OpenAI-compatible VLM server matching test_llm.py."""
        if not self.server_url:
            return self._heuristic_fallback(image, element_id)

        import httpx

        w, h = image.size
        # Downscale large image only if max_image_dim is set and image exceeds it
        if self.max_image_dim and max(w, h) > self.max_image_dim:
            scale = self.max_image_dim / max(w, h)
            proc_img = image.resize((int(w * scale), int(h * scale)), PILImage.Resampling.LANCZOS)
        else:
            proc_img = image

        # Encode image to PNG base64 matching test_llm.py (lossless, preserves transparency)
        buffered = io.BytesIO()
        proc_img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        mime_type = "image/png"

        # Match prompt in test_llm.py exactly (can be overridden via VLM_PROMPT env var)
        prompt_text = os.environ.get(
            "VLM_PROMPT",
            "Transcribe this image. If it contains a table or diagram, convert to markdown/mermaid.",
        )

        # EXACT payload matching test_llm.py
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
                                "url": f"data:{mime_type};base64,{img_b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # Print payload preview (truncating the huge base64 string for readability) matching test_llm.py
        preview_payload = json.loads(json.dumps(payload))
        preview_payload["messages"][0]["content"][1]["image_url"]["url"] = (
            f"data:{mime_type};base64," + img_b64[:30] + "...[truncated]..."
        )
        print("\n" + "=" * 60)
        print(f"🖼️  Image: {element_id} ({w}x{h} px)")
        print(f"📝 Prompt: {prompt_text}")
        print("Sending Payload:\n", json.dumps(preview_payload, indent=2))

        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            # trust_env=False bypasses proxy completely, exactly matching test_llm.py / curl --noproxy
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                print(f"\nCalling {self.server_url} (timeout {self.timeout}s, no proxy)...")
                resp = await client.post(self.server_url, json=payload, headers=headers)

                print("\nStatus Code:", resp.status_code)
                if resp.status_code == 200:
                    result = resp.json()
                    print("\nFull Response JSON:\n", json.dumps(result, indent=2, ensure_ascii=False))
                    choices = result.get("choices", [])
                    raw_content = ""
                    if choices and isinstance(choices, list):
                        raw_content = choices[0].get("message", {}).get("content", "")
                    print("\n=== Model Output ===\n", raw_content)
                    print("=" * 60 + "\n")

                    if raw_content:
                        return self._parse_vlm_output(raw_content, element_id)
                else:
                    print("Error Response:\n", resp.text)
                    print("=" * 60 + "\n")
        except Exception as e:
            print("Request failed:", e)
            print("=" * 60 + "\n")

        return self._heuristic_fallback(image, element_id)

    def _parse_vlm_output(self, raw_content: str, element_id: str) -> VLMProcessResult:
        """Parse structured fields from local VLM response (JSON, Mermaid, Table, or Markdown)."""
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
                raw_explanation=parsed.get("description") or parsed.get("raw_explanation") or text,
                confidence=0.95,
            )

        # 3. Check for Mermaid diagram
        mermaid_match = re.search(r"```mermaid\s*(.*?)\s*```", text, re.DOTALL)
        if mermaid_match:
            return VLMProcessResult(
                element_id=element_id,
                classification=VLMClassificationLabel.FLOWCHART_DIAGRAM,
                mermaid_code=mermaid_match.group(1).strip(),
                raw_explanation=text,
                confidence=0.95,
            )

        # 4. Check for Markdown table
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
        if len(table_lines) >= 2:
            return VLMProcessResult(
                element_id=element_id,
                classification=VLMClassificationLabel.TABLE_IMAGE,
                markdown_table="\n".join(table_lines),
                raw_explanation=text,
                confidence=0.95,
            )

        # 5. Direct Markdown / Text transcription
        return VLMProcessResult(
            element_id=element_id,
            classification=VLMClassificationLabel.OTHER,
            raw_explanation=text,
            confidence=0.95,
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

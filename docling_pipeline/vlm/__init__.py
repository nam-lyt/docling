"""VLM processing, safe annotation hooking, and custom markdown serialization."""

from docling_pipeline.vlm.annotation_hook import (
    AsyncVLMWorkerPool,
    VLMClassificationLabel,
    VLMProcessResult,
    attach_vlm_result,
    export_hybrid_markdown,
)

__all__ = [
    "AsyncVLMWorkerPool",
    "VLMClassificationLabel",
    "VLMProcessResult",
    "attach_vlm_result",
    "export_hybrid_markdown",
]

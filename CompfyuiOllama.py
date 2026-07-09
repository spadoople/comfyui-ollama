from __future__ import annotations
import copy
import random
import re
import json
import time
from contextlib import nullcontext
from urllib.parse import urlparse
import ipaddress

from ollama import Client
import numpy as np
import base64
from io import BytesIO
from server import PromptServer
from aiohttp import web
from pprint import pprint
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import os
from typing import TYPE_CHECKING, Any, Literal
from dataclasses import dataclass, field
from pydantic.json_schema import JsonSchemaValue


# --- Phoenix tracing helpers (Codex, local-content-safe) ---
_PHOENIX_TRACER = None
_PHOENIX_INIT_ATTEMPTED = False


def _phoenix_truthy_env(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off", ""}


def _phoenix_endpoint() -> str | None:
    return os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")


def _phoenix_endpoint_is_local() -> bool:
    endpoint = _phoenix_endpoint()
    if not endpoint:
        # Phoenix's helper defaults to localhost when no endpoint is supplied.
        return True
    try:
        host = urlparse(endpoint).hostname or ""
    except Exception:
        return False

    host_l = host.lower()
    if host_l in {"localhost", "127.0.0.1", "::1"}:
        return True

    # Treat RFC1918/private LAN collector IPs as local for homelab tracing.
    # This allows full prompt/response capture to a private Phoenix collector
    # such as 192.168.x.x without permitting arbitrary public endpoints.
    try:
        ip = ipaddress.ip_address(host_l.strip("[]"))
        return ip.is_private or ip.is_loopback
    except ValueError:
        return False


def _get_phoenix_tracer():
    """Return a Phoenix/OpenInference tracer, or None if tracing is unavailable.

    This is lazy and fail-open so ComfyUI generation continues if Phoenix is
    down, unconfigured, or the optional package is missing.
    """
    global _PHOENIX_TRACER, _PHOENIX_INIT_ATTEMPTED
    if _PHOENIX_TRACER is not None:
        return _PHOENIX_TRACER
    if _PHOENIX_INIT_ATTEMPTED:
        return None
    _PHOENIX_INIT_ATTEMPTED = True

    if not _phoenix_truthy_env("PHOENIX_ENABLED", "1"):
        return None

    try:
        from phoenix.otel import register

        kwargs = {
            "project_name": os.getenv("PHOENIX_PROJECT_NAME", "comfyui-ollama"),
            "batch": _phoenix_truthy_env("PHOENIX_BATCH", "0"),
            "auto_instrument": False,
            "verbose": _phoenix_truthy_env("PHOENIX_VERBOSE", "0"),
        }
        endpoint = _phoenix_endpoint()
        api_key = os.getenv("PHOENIX_API_KEY")
        if endpoint:
            kwargs["endpoint"] = endpoint
        if api_key:
            kwargs["api_key"] = api_key

        provider = register(**kwargs)
        _PHOENIX_TRACER = provider.get_tracer("comfyui-ollama")
        content_mode = "full local content" if _phoenix_capture_content() else "metadata only"
        auth_mode = "present" if api_key else "missing"
        print(f"[comfyui-ollama] Phoenix tracing enabled: project={kwargs['project_name']} endpoint={endpoint or 'default-local'} mode={content_mode} auth={auth_mode} batch={kwargs['batch']}")
        return _PHOENIX_TRACER
    except Exception as exc:
        print(f"[comfyui-ollama] Phoenix tracing disabled: {exc}")
        return None


def _phoenix_capture_content() -> bool:
    # Full prompt/response text is only emitted to local collectors. Remote
    # endpoints receive metadata only.
    return _phoenix_truthy_env("PHOENIX_CAPTURE_CONTENT", "1") and _phoenix_endpoint_is_local()


def _phoenix_start_span(name: str):
    tracer = _get_phoenix_tracer()
    if tracer is None:
        return nullcontext(None)
    return tracer.start_as_current_span(name)


def _phoenix_set(span, key: str, value) -> None:
    if span is None or value is None:
        return
    try:
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        span.set_attribute(key, value)
    except Exception:
        pass


def _phoenix_set_text(span, key: str, value, limit: int | None = None) -> None:
    if value is None or not _phoenix_capture_content():
        return
    text_value = str(value)
    if limit is None:
        try:
            limit = int(os.getenv("PHOENIX_CONTENT_LIMIT", "20000"))
        except Exception:
            limit = 20000
    if limit > 0 and len(text_value) > limit:
        text_value = text_value[:limit] + f"\n...[truncated {len(str(value)) - limit} chars]"
    _phoenix_set(span, key, text_value)


def _phoenix_mark_error(span, exc: BaseException) -> None:
    _phoenix_set(span, "error", True)
    _phoenix_set(span, "exception.type", exc.__class__.__name__)
    _phoenix_set(span, "exception.message", str(exc))
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, str(exc)))
        span.record_exception(exc)
    except Exception:
        pass


def _phoenix_coerce_mapping(value) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    return None


def _phoenix_extract_generate_text(response) -> str | None:
    data = _phoenix_coerce_mapping(response)
    if data is not None:
        text = data.get("response")
        return None if text is None else str(text)
    if hasattr(response, "response"):
        text = getattr(response, "response", None)
        return None if text is None else str(text)
    return str(response) if response is not None else None


def _phoenix_extract_generate_thinking(response) -> str | None:
    data = _phoenix_coerce_mapping(response)
    if data is not None:
        thinking = data.get("thinking")
        return None if thinking is None else str(thinking)
    if hasattr(response, "thinking"):
        thinking = getattr(response, "thinking", None)
        return None if thinking is None else str(thinking)
    return None


def _phoenix_extract_chat_text(response) -> str | None:
    if hasattr(response, "message"):
        message = response.message
        content = getattr(message, "content", None)
        return None if content is None else str(content)
    data = _phoenix_coerce_mapping(response)
    if data is not None:
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            return None if content is None else str(content)
    return str(response) if response is not None else None


def _phoenix_extract_chat_thinking(response) -> str | None:
    if hasattr(response, "message"):
        thinking = getattr(response.message, "thinking", None)
        return None if thinking is None else str(thinking)
    data = _phoenix_coerce_mapping(response)
    if data is not None:
        message = data.get("message")
        if isinstance(message, dict):
            thinking = message.get("thinking")
            return None if thinking is None else str(thinking)
    return None


def _phoenix_extract_token_counts(response) -> tuple[int | None, int | None]:
    data = _phoenix_coerce_mapping(response)
    if data is None:
        return None, None
    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    return (
        int(prompt_tokens) if prompt_tokens is not None else None,
        int(completion_tokens) if completion_tokens is not None else None,
    )


def _phoenix_build_generate_messages(system: str | None, prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if prompt:
        messages.append({"role": "user", "content": prompt})
    return messages


def _phoenix_sanitize_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    if not messages:
        return sanitized
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if content is None:
            content = ""
        text = str(content)
        images = message.get("images")
        if images:
            text = f"{text}\n[{len(images)} image(s) attached]"
        if role:
            sanitized.append({"role": str(role), "content": text})
    return sanitized


def _phoenix_apply_openinference_helpers():
    try:
        from openinference.instrumentation._attributes import (
            get_llm_input_message_attributes,
            get_llm_output_message_attributes,
        )
        from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

        return {
            "get_input_attrs": get_llm_input_message_attributes,
            "get_output_attrs": get_llm_output_message_attributes,
            "span_kind_attr": SpanAttributes.OPENINFERENCE_SPAN_KIND,
            "span_kind_llm": OpenInferenceSpanKindValues.LLM.value,
            "model_name_attr": SpanAttributes.LLM_MODEL_NAME,
            "invocation_params_attr": SpanAttributes.LLM_INVOCATION_PARAMETERS,
            "token_prompt_attr": SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
            "token_completion_attr": SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
            "token_total_attr": SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
            "input_value_attr": SpanAttributes.INPUT_VALUE,
            "output_value_attr": SpanAttributes.OUTPUT_VALUE,
            "session_id_attr": SpanAttributes.SESSION_ID,
            "tags_attr": SpanAttributes.TAG_TAGS,
            "metadata_attr": SpanAttributes.METADATA,
        }
    except Exception:
        return None


def _phoenix_begin_llm_span(span, model_name: str | None) -> None:
    helpers = _phoenix_apply_openinference_helpers()
    if helpers is None:
        _phoenix_set(span, "openinference.span.kind", "LLM")
        _phoenix_set(span, "llm.model_name", model_name)
        return
    _phoenix_set(span, helpers["span_kind_attr"], helpers["span_kind_llm"])
    _phoenix_set(span, helpers["model_name_attr"], model_name)


def _phoenix_record_llm_messages(
    span,
    input_messages: list[dict[str, str]],
    output_text: str | None,
    thinking: str | None = None,
) -> None:
    helpers = _phoenix_apply_openinference_helpers()
    if helpers is None:
        for index, message in enumerate(input_messages):
            _phoenix_set_text(span, f"llm.input_messages.{index}.message.role", message.get("role"))
            _phoenix_set_text(span, f"llm.input_messages.{index}.message.content", message.get("content"))
        if output_text is not None:
            _phoenix_set_text(span, "llm.output_messages.0.message.role", "assistant")
            _phoenix_set_text(span, "llm.output_messages.0.message.content", output_text)
        if thinking:
            _phoenix_set_text(span, "llm.output.thinking", thinking)
        return

    for key, value in helpers["get_input_attrs"](input_messages).items():
        _phoenix_set_text(span, key, value)

    output_messages: list[dict[str, str]] = []
    if output_text is not None:
        output_messages.append({"role": "assistant", "content": output_text})
    for key, value in helpers["get_output_attrs"](output_messages).items():
        _phoenix_set_text(span, key, value)

    if thinking:
        _phoenix_set_text(span, "llm.output.thinking", thinking)

    if input_messages:
        _phoenix_set_text(span, helpers["input_value_attr"], json.dumps(input_messages, ensure_ascii=False))
    if output_text is not None:
        _phoenix_set_text(span, helpers["output_value_attr"], output_text)


def _phoenix_record_ollama_result(
    span,
    response,
    duration_ms: float,
    request_options: dict[str, Any] | None,
) -> None:
    helpers = _phoenix_apply_openinference_helpers()
    prompt_tokens, completion_tokens = _phoenix_extract_token_counts(response)
    if helpers is not None:
        if request_options:
            _phoenix_set(span, helpers["invocation_params_attr"], request_options)
        if prompt_tokens is not None:
            _phoenix_set(span, helpers["token_prompt_attr"], prompt_tokens)
        if completion_tokens is not None:
            _phoenix_set(span, helpers["token_completion_attr"], completion_tokens)
        if prompt_tokens is not None and completion_tokens is not None:
            _phoenix_set(span, helpers["token_total_attr"], prompt_tokens + completion_tokens)
    else:
        if request_options:
            _phoenix_set(span, "llm.invocation_parameters", request_options)
        if prompt_tokens is not None:
            _phoenix_set(span, "llm.token_count.prompt", prompt_tokens)
        if completion_tokens is not None:
            _phoenix_set(span, "llm.token_count.completion", completion_tokens)

    _phoenix_set(span, "llm.duration_ms", duration_ms)

    data = _phoenix_coerce_mapping(response)
    if data is not None:
        for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration", "done_reason"):
            if data.get(key) is not None:
                _phoenix_set(span, f"ollama.{key}", data.get(key))


def _phoenix_mark_ok(span) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.OK))
    except Exception:
        pass


def _phoenix_session_prefix() -> str:
    return os.getenv("PHOENIX_SESSION_PREFIX", "").strip().strip(":")


def _phoenix_extra_tags() -> list[str]:
    raw = os.getenv("PHOENIX_TAGS", "").strip()
    if not raw:
        return []
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def _phoenix_sanitize_tag_value(value: str) -> str:
    cleaned = re.sub(r"[^\w.\-/]+", "-", str(value).strip())
    return cleaned[:128] if cleaned else "unknown"


def _phoenix_build_tags(*, node_type: str, model_name: str | None) -> list[str]:
    tags = [f"node-type:{node_type}"]
    if model_name:
        tags.append(f"model:{_phoenix_sanitize_tag_value(model_name)}")
    prompt_id, _ = _phoenix_get_execution_context()
    if prompt_id:
        tags.append(f"workflow:{prompt_id}")
    tags.extend(_phoenix_extra_tags())
    # Preserve order but drop duplicates.
    return list(dict.fromkeys(tags))


def _phoenix_build_trace_metadata(
    *,
    node_type: str,
    model_name: str | None,
    chat_session_key: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"comfy.node.type": node_type}
    if model_name:
        metadata["llm.model_name"] = model_name
    prompt_id, node_id = _phoenix_get_execution_context()
    if prompt_id:
        metadata["comfy.prompt_id"] = prompt_id
    if node_id:
        metadata["comfy.node_id"] = node_id
    if chat_session_key:
        metadata["ollama.session_key"] = chat_session_key
    prefix = _phoenix_session_prefix()
    if prefix:
        metadata["phoenix.session_prefix"] = prefix
    return metadata


def _phoenix_get_execution_context() -> tuple[str | None, str | None]:
    try:
        from comfy_execution.utils import get_executing_context

        ctx = get_executing_context()
        if ctx is None:
            return None, None
        return ctx.prompt_id, ctx.node_id
    except Exception:
        return None, None


def _phoenix_resolve_session_id(*, chat_session_key: str | None = None) -> str | None:
    session_id: str | None
    if chat_session_key:
        session_id = f"ollama-chat:{chat_session_key}"
    else:
        prompt_id, _ = _phoenix_get_execution_context()
        session_id = f"comfyui-prompt:{prompt_id}" if prompt_id else None
    prefix = _phoenix_session_prefix()
    if session_id and prefix:
        session_id = f"{prefix}:{session_id}"
    return session_id


def _phoenix_trace_context(
    session_id: str | None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    if not session_id and not tags and not metadata:
        return nullcontext(None)
    try:
        from phoenix.otel import using_attributes

        kwargs: dict[str, Any] = {}
        if session_id:
            kwargs["session_id"] = session_id
        if tags:
            kwargs["tags"] = tags
        if metadata:
            kwargs["metadata"] = metadata
        return using_attributes(**kwargs)
    except Exception:
        return _phoenix_session_context(session_id)


def _phoenix_session_context(session_id: str | None):
    if not session_id:
        return nullcontext(None)
    try:
        from phoenix.otel import using_session

        return using_session(session_id=session_id)
    except Exception:
        return nullcontext(None)


def _phoenix_apply_trace_attrs(
    span,
    session_id: str | None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    _phoenix_apply_session_id(span, session_id)
    helpers = _phoenix_apply_openinference_helpers()
    if tags:
        if helpers is not None:
            _phoenix_set(span, helpers["tags_attr"], tags)
        else:
            _phoenix_set(span, "tag.tags", tags)
    if metadata:
        if helpers is not None:
            _phoenix_set(span, helpers["metadata_attr"], metadata)
        else:
            _phoenix_set(span, "metadata", metadata)


def _phoenix_apply_session_id(span, session_id: str | None) -> None:
    if not session_id:
        return
    helpers = _phoenix_apply_openinference_helpers()
    if helpers is not None:
        _phoenix_set(span, helpers["session_id_attr"], session_id)
    else:
        _phoenix_set(span, "session.id", session_id)


def _phoenix_apply_comfy_execution_attrs(span) -> None:
    prompt_id, node_id = _phoenix_get_execution_context()
    if prompt_id:
        _phoenix_set(span, "comfy.prompt_id", prompt_id)
    if node_id:
        _phoenix_set(span, "comfy.node_id", node_id)
# --- End Phoenix tracing helpers (Codex) ---

# For type checking only. Torch is not installed at runtime
if TYPE_CHECKING:
    import torch


@dataclass
class ChatSession:
    messages: list[dict] = field(default_factory=list)
    model: str = ""


# Dictionary global per session_id
CHAT_SESSIONS: dict[str, ChatSession] = {}

# Function to filter enabled options
def _filter_enabled_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return only the ollama options whose 'enable_*' flag is True."""
    if not options:
        return None
    enablers = [
        "enable_mirostat",
        "enable_mirostat_eta",
        "enable_mirostat_tau",
        "enable_num_ctx",
        "enable_repeat_last_n",
        "enable_repeat_penalty",
        "enable_temperature",
        "enable_seed",
        "enable_stop",
        "enable_tfs_z",
        "enable_num_predict",
        "enable_top_k",
        "enable_top_p",
        "enable_min_p",
    ]
    out: dict[str, Any] = {}
    for enabler in enablers:
        if options.get(enabler, False):
            key = enabler.replace("enable_", "")
            out[key] = options[key]
    return out or None

@PromptServer.instance.routes.post("/ollama/get_models")
async def get_models_endpoint(request):
    data = await request.json()

    url = data.get("url")
    client = Client(host=url)

    models = client.list().get('models', [])

    try:
        models = [model['model'] for model in models]
        return web.json_response(models)
    except Exception as e:
        models = [model['name'] for model in models]
        return web.json_response(models)

class OllamaSaveContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"context": ("STRING", {"forceInput": True},),
                     "filename": ("STRING", {"default": "context"})},
                }

    RETURN_TYPES = ()
    FUNCTION = "ollama_save_context"

    OUTPUT_NODE = True
    CATEGORY = "Ollama"

    def ollama_save_context(self, filename, context=None):
        path = self._base_dir + os.path.sep + filename
        metadata = PngInfo()

        metadata.add_text("context", ','.join(map(str, context)))

        image = Image.new('RGB', (100, 100), (255, 255, 255))  # Creates a 100x100 white image

        image.save(path + ".png", pnginfo=metadata)

        return {"ui": {"context": context}}


class OllamaLoadContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f != ".keep"]
        return {"required":
                    {"context_file": (files, {})},
                }

    CATEGORY = "Ollama"

    RETURN_NAMES = ("context",)
    RETURN_TYPES = ("STRING",)
    FUNCTION = "ollama_load_context"

    def ollama_load_context(self, context_file):
        with Image.open(self._base_dir + os.path.sep + context_file) as img:
            info = img.info
            res = info.get('context', '')
        return (res,)


class OllamaOptionsV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "enable_mirostat": ("BOOLEAN", {"default": False}),
                "mirostat": ("INT", {"default": 0, "min": 0, "max":2, "step": 1, "tooltip": "Whether to use Mirostat sampling. Mirostat is an algorithm that actively maintains the quality of generated text within a desired range during text generation. (0 = disabled, 1 = Mirostat 1, 2 = Mirostat 2.0)"}),

                "enable_mirostat_eta": ("BOOLEAN", {"default": False}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0, "step": 0.1, "tooltip": "Mirostat's learning rate parameter influences how quickly the algorithm responds to feedback from the generated text."}),

                "enable_mirostat_tau": ("BOOLEAN", {"default": False}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0, "step": 0.1, "tooltip": "Mirostat's target entropy parameter controls the balance between coherence and diversity in the generated text."}),

                "enable_num_ctx": ("BOOLEAN", {"default": False}),
                "num_ctx": ("INT", {"default": 2048, "min": 0, "max": 2 ** 31, "step": 1, "tooltip": "Sets the size of the context window used to generate the next token."}),

                "enable_repeat_last_n": ("BOOLEAN", {"default": False}),
                "repeat_last_n": ("INT", {"default": 64, "min": -1, "max": 64, "step": 1, "tooltip": "Sets how far back for the model to look back to prevent repetition. (0 = disabled, -1 = num_ctx)"}),

                "enable_repeat_penalty": ("BOOLEAN", {"default": False}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0, "max": 2, "step": 0.05, "tooltip": "Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient."}),

                "enable_temperature": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.8, "min": -10, "max": 10, "step": 0.05, "tooltip": "Increasing the temperature will make the model answer more creatively."}),

                "enable_seed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1, "tooltip": "Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt."}),

                "enable_stop": ("BOOLEAN", {"default": False}),
                "stop": ("STRING", {"default": "", "multiline": False, "tooltip": "When this pattern is encountered the LLM will stop generating text and return."}),

                "enable_tfs_z": ("BOOLEAN", {"default": False}),
                "tfs_z": ("FLOAT", {"default": 1, "min": 1, "max": 1000, "step": 0.05}),

                "enable_num_predict": ("BOOLEAN", {"default": False}),
                "num_predict": ("INT", {"default": -1, "min": -2, "max": 2048, "step": 1, "tooltip": "Maximum number of tokens to predict when generating text. The default -1 means infinite generation."}),

                "enable_top_k": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1, "tooltip": "Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative."}),

                "enable_top_p": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05, "tooltip": "Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text."}),

                "enable_min_p": ("BOOLEAN", {"default": False}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05, "tooltip": "Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out."}),

                "debug": ("BOOLEAN", {"default": False, "tooltip": "For debugging purposes of the custom nodes, no effect on ollama api."}),
            },
        }

    RETURN_TYPES = ("OLLAMA_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "ollama_options"
    CATEGORY = "Ollama"
    DESCRIPTION = "Various settings for advanced configuration of Ollama inference. See Ollama documentation for more details."

    def ollama_options(self, **kargs):

        if kargs['debug']:
            print("--- ollama options v2 dump\n")
            pprint(kargs)
            print("---------------------------------------------------------")

        return (kargs,)

class OllamaConnectivityV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "The URL of the Ollama server. Default value points to a local instance with ollama's default port configuration."
                }),
                "model": ((), {"tooltip": "Select a model for inference. This is a list of available models on the Ollama server. If you don't see any, make sure the Ollama server is running on the url and there are models installed."}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 120, "step": 1, "tooltip": "Configures how long ollama keeps the model loaded in memory after inference. -1 = keep alive indefinitely, 0 = unload model immediately after inference"}),
                "keep_alive_unit": (["minutes", "hours"],),
            },
        }

    RETURN_TYPES = ("OLLAMA_CONNECTIVITY",)
    RETURN_NAMES = ("connection",)
    FUNCTION = "ollama_connectivity"
    CATEGORY = "Ollama"
    DESCRIPTION = "Provides connection to an Ollama server. Use the refresh button to load the model list in case of connection error or after installing a new model."

    def ollama_connectivity(self, url, model, keep_alive, keep_alive_unit):
        data = {
            "url": url,
            "model": model,
            "keep_alive": keep_alive,
            "keep_alive_unit": keep_alive_unit,
        }

        return (data,)


class OllamaGenerateV2:
    def __init__(self):
        self.saved_context = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an AI artist.",
                    "tooltip": "System prompt - use this to set the role and general behavior of the model."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is art?",
                    "tooltip": "User prompt - a question or task you want the model to answer or perform. For vision tasks, you can refer to the input image as 'this image', 'photo' etc. like 'Describe this image in detail'"
                }),
                "think": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will do a thinking process before answering. This can result in more accurate results. The thinking is then available as a separate output for debugging or understanding how the model arrived at its answer. Some models don't support this feature and the generation will fail."}),
                "keep_context": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will keep the context of the conversation and use it for the next generation. This is useful for multi-turn conversations or tasks that require context."}),
                "format": (["text", "json"], {"tooltip": "Output format of the response. 'text' will return a plain text response, while 'json' will return a structured response in JSON format. This is useful when the model is part of a larger pipeline and you need additional processing on the response. In this case I recommend showing the model example outputs in the system prompt. Some models are not trained to perform well in structured output."}),

            },
            "optional": {
                "connectivity": ("OLLAMA_CONNECTIVITY", {"forceInput": False, "tooltip": "Set an ollama provider for the generation. If this input is empty, the 'meta' input must be set."},),
                "options": ("OLLAMA_OPTIONS", {"forceInput": False, "tooltip": "Connect an Ollama Options node for advanced inference configuration."},),
                "images": ("IMAGE", {"forceInput": False, "tooltip": "Provide an image or a batch of images for vision tasks. Make sure that the selected model supports vision, otherwise it may hallucinate the response."},),
                "context": ("OLLAMA_CONTEXT", {"forceInput": False, "tooltip": "Optionally set an existing model context, useful for multi-turn conversations, follow-up questions."},),
                "meta": ("OLLAMA_META", {"forceInput": False, "tooltip": "Use this input to chain multiple 'Ollama Generate' nodes. In this case the connectivity and options inputs are passed along."},),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "OLLAMA_CONTEXT", "OLLAMA_META",)
    RETURN_NAMES = ("result", "thinking", "context", "meta",)
    FUNCTION = "ollama_generate_v2"
    CATEGORY = "Ollama"
    DESCRIPTION = "Text generation with Ollama. Supports vision tasks, multi-turn conversations, and advanced inference options. Connect an Ollama Connectivity node to set the server URL and model."

    def get_request_options(self, options):
        response = None

        if options is None:
            return response

        enablers = ['enable_mirostat', 'enable_mirostat_eta',
                    'enable_mirostat_tau', 'enable_mirostat_eta',
                    'enable_num_ctx', 'enable_repeat_last_n', 'enable_repeat_penalty',
                    'enable_temperature', 'enable_seed', 'enable_stop', 'enable_tfs_z', 'enable_num_predict',
                    'enable_top_k', 'enable_top_p', 'enable_min_p']

        for enabler in enablers:
            if options[enabler]:
                if response is None:
                    response = {}
                key = enabler.replace("enable_", "")
                response[key] = options[key]

        return response

    def ollama_generate_v2(self, system, prompt, think, keep_context, format, context = None, options=None, connectivity=None, images=None, meta=None):

        if connectivity is None and meta is None:
            raise Exception("Required input connectivity or meta.")

        if connectivity is None and meta['connectivity'] is None:
            raise Exception("Required input connectivity or connectivity in meta.")

        if meta is not None:
            if connectivity is not None: # bypass the current meta connectivity
                meta["connectivity"] = connectivity
            if options is not None: # bypass the current meta options
                meta["options"] = options
        else:
            meta = {"options": options, "connectivity": connectivity}

        url = meta['connectivity']['url']
        model = meta['connectivity']['model']
        client = Client(host=url)

        debug_print = True if meta['options'] is not None and meta['options']['debug'] else False

        if format == "text":
            format = ''

        if context is not None and isinstance(context, str):
            string_list = context.split(',')
            context = [int(item.strip()) for item in string_list]

        if keep_context and context is None:
            context = self.saved_context

        keep_alive_unit =  'm' if meta['connectivity']['keep_alive_unit'] == "minutes" else 'h'
        request_keep_alive = str(meta['connectivity']['keep_alive']) + keep_alive_unit

        request_options = self.get_request_options(options)

        images_b64 = None
        if images is not None:
            images_b64 = []
            for (batch_number, image) in enumerate(images):
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = base64.b64encode(buffered.getvalue())
                images_b64.append(str(img_bytes, 'utf-8'))

        if debug_print:
            print(f"""
--- ollama generate v2 request: 

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {0 if images_b64 is None else len(images_b64)}
context: {context}
think: {think}
options: {request_options}
keep alive: {request_keep_alive}
format: {format}
---------------------------------------------------------
""")

        phoenix_attrs = {
            "comfy.node.type": "OllamaGenerateV2",
            "llm.provider": "ollama",
            "llm.system": "ollama",
            "llm.model_name": model,
            "ollama.url": url,
            "ollama.keep_alive": request_keep_alive,
            "ollama.format": format,
            "ollama.think": bool(think),
            "ollama.has_context": context is not None,
            "ollama.image_count": 0 if images_b64 is None else len(images_b64),
            "input.mime_type": "text/plain",
            "phoenix.content_capture": _phoenix_capture_content(),
        }
        session_id = _phoenix_resolve_session_id()
        trace_tags = _phoenix_build_tags(node_type="OllamaGenerateV2", model_name=model)
        trace_metadata = _phoenix_build_trace_metadata(
            node_type="OllamaGenerateV2",
            model_name=model,
        )
        with _phoenix_trace_context(session_id, trace_tags, trace_metadata):
            with _phoenix_start_span("comfyui.ollama.generate") as span:
                _phoenix_apply_trace_attrs(span, session_id, trace_tags, trace_metadata)
                _phoenix_apply_comfy_execution_attrs(span)
                for key, value in phoenix_attrs.items():
                    _phoenix_set(span, key, value)
                input_messages = _phoenix_build_generate_messages(system, prompt)
                _phoenix_begin_llm_span(span, model)
                _phoenix_record_llm_messages(span, input_messages, None)
                started_at = time.perf_counter()
                try:
                    response = client.generate(
                        model=model,
                        system=system,
                        prompt=prompt,
                        images=images_b64,
                        context=context,
                        think=think,
                        options=request_options,
                        keep_alive= request_keep_alive,
                        format=format,
                    )
                except Exception as exc:
                    _phoenix_mark_error(span, exc)
                    raise
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                response_text = _phoenix_extract_generate_text(response)
                thinking_text = _phoenix_extract_generate_thinking(response) if think else None
                _phoenix_record_ollama_result(span, response, duration_ms, request_options)
                _phoenix_record_llm_messages(span, input_messages, response_text, thinking_text)
                _phoenix_mark_ok(span)

        if debug_print:
            print("\n--- ollama generate v2 response:")
            pprint(response)
            print("---------------------------------------------------------")

        ollama_response_text = response['response']
        ollama_response_thinking = response['thinking'] if think else None

        if keep_context:
            self.saved_context = response["context"]
            if debug_print:
                print("saving context to node memory.")

        return ollama_response_text, ollama_response_thinking, response['context'], meta,


class OllamaChat:
    """
    Text generation with Ollama Chat.
    Returns: (result: str, thinking: str|None, meta: dict, history: str)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are an AI artist.",
                        "tooltip": "System prompt - use this to set the role and general behavior of the model.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "What is art?",
                        "tooltip": "User prompt - a question or task you want the model to answer or perform. For vision tasks, you can refer to the input image as 'this image', 'photo' etc. like 'Describe this image in detail'",
                    },
                ),
                "think": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, the model will do a thinking process before answering. This can result in more accurate results. The thinking is then available as a separate output for debugging or understanding how the model arrived at its answer. Some models don't support this feature and the generation will fail.",
                    },
                ),
                "format": (
                    ["text", "json"],
                    {
                        "tooltip": "Output format of the response. 'text' will return a plain text response, while 'json' will return a structured response in JSON format. This is useful when the model is part of a larger pipeline and you need additional processing on the response. In this case I recommend showing the model example outputs in the system prompt. Some models are not trained to perform well in structured output."
                    },
                ),
            },
            "optional": {
                "connectivity": (
                    "OLLAMA_CONNECTIVITY",
                    {
                        "forceInput": False,
                        "tooltip": "Set an ollama provider for the generation. If this input is empty, the 'meta' input must be set.",
                    },
                ),
                "options": (
                    "OLLAMA_OPTIONS",
                    {
                        "forceInput": False,
                        "tooltip": "Connect an Ollama Options node for advanced inference configuration.",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "forceInput": False,
                        "tooltip": "Provide an image or a batch of images for vision tasks. Make sure that the selected model supports vision, otherwise it may hallucinate the response.",
                    },
                ),
                "meta": (
                    "OLLAMA_META",
                    {
                        "forceInput": False,
                        "tooltip": "Use this input to chain multiple 'Ollama Generate' nodes. In this case the connectivity and options inputs are passed along.",
                    },
                ),
                "history": (
                    "OLLAMA_HISTORY",
                    {
                        "forceInput": False,
                        "tooltip": "Optionally set an existing model history, useful for multi-turn conversations, follow-up questions.",
                    },
                ),
                "reset_session": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Clear the conversation history. WARNING: If using shared history, this will affect all nodes using the same history ID.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "OLLAMA_META",
        "OLLAMA_HISTORY",
    )
    RETURN_NAMES = (
        "result",
        "thinking",
        "meta",
        "history",
    )
    FUNCTION = "ollama_chat"
    CATEGORY = "Ollama"
    DESCRIPTION = "Text generation with Ollama Chat. Supports vision tasks, multi-turn conversations, and advanced inference options. Connect an Ollama Connectivity node to set the server URL and model."

    def ollama_chat(
        self,
        system: str,
        prompt: str,
        think: bool,
        unique_id: str,
        format: str,
        options: dict[str, Any] | None = None,
        connectivity: dict[str, Any] | None = None,
        images: list[torch.Tensor] | None = None,
        meta: dict[str, Any] | None = None,
        history: str | None = None,
        reset_session: bool = False,
    ) -> tuple[str | None, str | None, dict[str, Any], str | None]:

        if meta is None:
            if connectivity is None:
                raise ValueError("Either 'connectivity' or 'meta' must be provided.")
            meta = {}

        # Update with provided values (override)
        if connectivity is not None:
            meta["connectivity"] = connectivity
        if options is not None:
            meta["options"] = options
        else:
            meta["options"] = None

        # Final validation
        if "connectivity" not in meta or meta["connectivity"] is None:
            raise ValueError("'connectivity' must be present in meta.")

        url = meta["connectivity"]["url"]
        model = meta["connectivity"]["model"]
        client = Client(host=url)

        debug_print = (
            True if meta["options"] is not None and meta["options"]["debug"] else False
        )

        ollama_format: Literal["", "json"] | JsonSchemaValue | None = None

        if format == "json":
            ollama_format = "json"
        elif format == "text":
            ollama_format = ""

        keep_alive_unit = (
            "m" if meta["connectivity"]["keep_alive_unit"] == "minutes" else "h"
        )
        request_keep_alive = str(meta["connectivity"]["keep_alive"]) + keep_alive_unit

        # 4. use the shared helper instead of self.get_request_options
        request_options = _filter_enabled_options(options)

        images_b64: list[str] | None = None
        if images is not None:
            images_b64 = []
            for batch_number, image in enumerate(images):
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images_b64.append(img_bytes)

        if debug_print:
            print(
                f"""
--- ollama chat request: 

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {0 if images_b64 is None else len(images_b64)}
think: {think}
options: {request_options}
keep alive: {request_keep_alive}
format: {format}
---------------------------------------------------------
"""
            )

        # Determinate which session to use
        session_key = history if history is not None else unique_id

        # If reset_session is True, reset the session
        if reset_session:
            CHAT_SESSIONS[session_key] = ChatSession()
            if debug_print:
                print(f"Session {session_key} has been reset")

        # If the session doesn't exist, create it
        if session_key not in CHAT_SESSIONS:
            CHAT_SESSIONS[session_key] = ChatSession()

        session = CHAT_SESSIONS[session_key]

        # Update history for return
        history = session_key

        # If there is a system prompt, replace it or add it to the beginning
        if system:
            if session.messages and session.messages[0].get("role") == "system":
                session.messages[0] = {"role": "system", "content": system}
            else:
                session.messages.insert(0, {"role": "system", "content": system})

        # Construct the user message for history
        user_message_for_history: dict[str, Any] = {
            "role": "user",
            "content": prompt,
        }

        # Add the user message to the history (without images)
        session.messages.append(user_message_for_history)

        if debug_print:
            print("\n--- ollama chat session:")
            for message in session.messages:
                pprint(f"{message['role']}> {message['content'][:50]}...")
                if "images" in message:
                    for image in message["images"]:
                        pprint(f"Image: {image[:50]}...")
            print("---------------------------------------------------------")

        # Construct the messages for the API call (with images)
        messages_for_api = copy.deepcopy(session.messages)

        # If there are images, modify the last user message for the API call
        if images_b64 is not None:
            messages_for_api[-1]["images"] = images_b64

        phoenix_attrs = {
            "comfy.node.type": "OllamaChat",
            "llm.provider": "ollama",
            "llm.system": "ollama",
            "llm.model_name": model,
            "ollama.url": url,
            "ollama.keep_alive": request_keep_alive,
            "ollama.format": ollama_format or "",
            "ollama.think": bool(think),
            "ollama.session_key": session_key,
            "ollama.message_count": len(messages_for_api),
            "ollama.image_count": 0 if images_b64 is None else len(images_b64),
            "input.mime_type": "text/plain",
            "phoenix.content_capture": _phoenix_capture_content(),
        }
        session_id = _phoenix_resolve_session_id(chat_session_key=session_key)
        trace_tags = _phoenix_build_tags(node_type="OllamaChat", model_name=model)
        trace_metadata = _phoenix_build_trace_metadata(
            node_type="OllamaChat",
            model_name=model,
            chat_session_key=session_key,
        )
        with _phoenix_trace_context(session_id, trace_tags, trace_metadata):
            with _phoenix_start_span("comfyui.ollama.chat") as span:
                _phoenix_apply_trace_attrs(span, session_id, trace_tags, trace_metadata)
                _phoenix_apply_comfy_execution_attrs(span)
                for key, value in phoenix_attrs.items():
                    _phoenix_set(span, key, value)
                input_messages = _phoenix_sanitize_messages(messages_for_api)
                _phoenix_begin_llm_span(span, model)
                _phoenix_record_llm_messages(span, input_messages, None)
                started_at = time.perf_counter()
                try:
                    response = client.chat(
                        model=model,
                        messages=messages_for_api,
                        options=request_options,
                        keep_alive=request_keep_alive,
                        format=ollama_format,
                    )
                except Exception as exc:
                    _phoenix_mark_error(span, exc)
                    raise
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                response_text = _phoenix_extract_chat_text(response)
                thinking_text = _phoenix_extract_chat_thinking(response) if think else None
                _phoenix_record_ollama_result(span, response, duration_ms, request_options)
                _phoenix_record_llm_messages(span, input_messages, response_text, thinking_text)
                _phoenix_mark_ok(span)

        if debug_print:
            print("\n--- ollama chat response:")
            pprint(response)
            print("---------------------------------------------------------")

        ollama_response_text = response.message.content
        ollama_response_thinking = response.message.thinking if think else None

        # Add the assistant message to the history
        session.messages.append(
            {
                "role": "assistant",
                "content": ollama_response_text,
            }
        )

        return (
            ollama_response_text,
            ollama_response_thinking,
            meta,
            history,
        )


NODE_CLASS_MAPPINGS = {
    "OllamaOptionsV2": OllamaOptionsV2,
    "OllamaConnectivityV2": OllamaConnectivityV2,
    "OllamaGenerateV2": OllamaGenerateV2,
    "OllamaSaveContext": OllamaSaveContext,
    "OllamaLoadContext": OllamaLoadContext,
    "OllamaChat": OllamaChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaOptionsV2": "Ollama Options",
    "OllamaConnectivityV2": "Ollama Connectivity",
    "OllamaGenerateV2": "Ollama Generate",
    "OllamaSaveContext": "Ollama Save Context",
    "OllamaLoadContext": "Ollama Load Context",
    "OllamaChat": "Ollama Chat",
}

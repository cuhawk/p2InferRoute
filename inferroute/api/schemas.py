"""Request validation models for the OpenAI-compatible API.

Non-standard extension fields are accepted at the top level and under ``route``:
``task``, ``route.strategy`` / ``route.min_context`` / ``route.privacy`` /
``route.capabilities``, and ``cache``. Unknown fields are preserved (``extra="allow"``)
and forwarded nowhere unless a documented feature consumes them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ALLOWED_ROLES = {"system", "user", "assistant", "tool", "developer", "function"}


class MessageModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    @field_validator("role")
    @classmethod
    def _valid_role(cls, v: str) -> str:
        if v not in ALLOWED_ROLES:
            raise ValueError(f"role must be one of {sorted(ALLOWED_ROLES)}")
        return v


class StreamOptionsModel(BaseModel):
    include_usage: bool = False


class RouteOptionsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str | None = None
    min_context: int | None = Field(default=None, ge=1)
    privacy: Literal["local", "private", "any"] | None = None
    capabilities: list[str] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "auto"
    messages: list[MessageModel] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    stream_options: StreamOptionsModel | None = None
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    user: str | None = None
    # ---- InferRoute extensions -------------------------------------------------
    task: str | None = None  # explicit task classification override
    route: RouteOptionsModel | None = None  # per-request routing controls
    cache: bool | None = None  # explicit cache bypass: false

    @field_validator("stop")
    @classmethod
    def _normalize_stop(cls, v: str | list[str] | None) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, v: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if v is None:
            return None
        for tool in v:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise ValueError("each tool must be an object with type == 'function'")
            if not isinstance(tool.get("function"), dict):
                raise ValueError("each tool must carry a 'function' object")
        return v

    def has_vision_content(self) -> bool:
        for msg in self.messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and str(part.get("type", "")).startswith("image"):
                        return True
        return False

    def to_raw_messages(self) -> list[dict[str, Any]]:
        return [
            {
                "role": m.role,
                **({"content": m.content} if m.content is not None else {}),
                **({"name": m.name} if m.name is not None else {}),
                **({"tool_calls": m.tool_calls} if m.tool_calls is not None else {}),
                **({"tool_call_id": m.tool_call_id} if m.tool_call_id is not None else {}),
            }
            for m in self.messages
        ]

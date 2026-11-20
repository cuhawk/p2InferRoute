"""Strict structured-output validation for the tiered cascade."""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry
from referencing.exceptions import Unresolvable

from inferroute.providers.base import ChatResult


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate JSON property")
    return obj


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def compile_schema(response_format: dict[str, Any]) -> Draft202012Validator:
    envelope = response_format["json_schema"]
    schema = envelope["schema"]
    if not isinstance(schema, dict):
        raise ValueError("schema must be a JSON object")
    if (
        schema.get("$schema", Draft202012Validator.META_SCHEMA["$id"])
        != (Draft202012Validator.META_SCHEMA["$id"])
    ):
        raise ValueError("expected JSON Schema Draft 2020-12")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=Registry(), format_checker=FormatChecker())


def valid_completion(result: ChatResult, validator: Draft202012Validator) -> bool:
    if not result.response.choices:
        return False
    for choice in result.response.choices:
        text = choice.message.content
        if not isinstance(text, str) or choice.finish_reason in {"length", "content_filter"}:
            return False
        try:
            obj = json.loads(
                text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
            if not validator.is_valid(obj):
                return False
        except (ValueError, RecursionError, Unresolvable):
            return False
    return True

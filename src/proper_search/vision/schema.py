"""Turning a Pydantic model into a provider-acceptable JSON schema.

Structured-output endpoints accept a restricted JSON Schema dialect. Pydantic
emits a richer one, so the extras are stripped here rather than at the call
site — a rejected schema fails the entire run, not one item, and the failure
message rarely names the offending keyword.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# Validation keywords the structured-output dialect does not accept. Pydantic
# emits several of these from ordinary field constraints.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "contentEncoding",
        "contentMediaType",
    }
)

# Carries no meaning for the model and costs tokens on every request.
_NOISE_KEYWORDS = frozenset({"title", "default"})


def to_api_schema(model: type[BaseModel], *, all_required: bool = True) -> dict[str, Any]:
    """Build a structured-output schema from a Pydantic model.

    ``all_required=True`` marks every property required even where the model
    has a default. Optional fields invite the model to omit them, and a
    description missing its ``on_screen_text`` is indistinguishable from one
    that genuinely had no text — a distinction the search index depends on.
    Pydantic defaults still apply when parsing, so nothing breaks if a provider
    omits a field anyway.
    """
    cleaned: dict[str, Any] = _clean(model.model_json_schema(), all_required=all_required)
    return cleaned


def _clean(node: Any, *, all_required: bool) -> Any:
    if isinstance(node, list):
        return [_clean(item, all_required=all_required) for item in node]
    if not isinstance(node, dict):
        return node

    is_object = node.get("type") == "object" and "properties" in node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_KEYWORDS or key in _NOISE_KEYWORDS:
            continue
        # Pydantic maps a model's *class docstring* onto the object node's
        # description. Those docstrings explain the implementation to future
        # maintainers ("these become individually embedded chunks"), which is
        # both wasted tokens and confusing prompt surface. Field-level
        # descriptions, written deliberately for the model, are kept.
        if key == "description" and is_object:
            continue
        out[key] = _clean(value, all_required=all_required)

    if is_object:
        # Required on every object, not just the root: a nested frame note with
        # an optional field has the same omission problem.
        out["additionalProperties"] = False
        if all_required:
            out["required"] = list(out["properties"].keys())
    return out


def describe_schema_size(schema: dict[str, Any]) -> int:
    """Rough character count, for reasoning about per-request token overhead."""
    import json

    return len(json.dumps(schema))

"""Argument coercion and the tool execution wrapper.

Nearly every case here is a shape a real model actually sent. The array ones
matter most: a researcher that cites its sources as a JSON string rather than
as a list has its findings rejected one by one, the run finishes with nothing
recorded, and the trace looks like the model simply failed to find anything.
"""
from __future__ import annotations

import pytest

from mas.kernel.contracts import ToolSpec
from mas.kernel.tool import ToolError, validate_arguments


def spec(properties: dict, required: list[str] | None = None) -> ToolSpec:
    return ToolSpec(
        name="t",
        description="d",
        parameters={"type": "object", "properties": properties, "required": required or []},
    )


ARRAY = spec({"refs": {"type": "array", "items": {"type": "string"}}}, ["refs"])


def test_a_proper_array_passes_through():
    assert validate_arguments(ARRAY, {"refs": ["S1", "S2"]})["refs"] == ["S1", "S2"]


def test_a_json_encoded_array_is_unwrapped():
    """The failure that cost a live run every one of its findings."""
    assert validate_arguments(ARRAY, {"refs": '["S4"]'})["refs"] == ["S4"]
    assert validate_arguments(ARRAY, {"refs": '["S1", "S2"]'})["refs"] == ["S1", "S2"]


def test_a_single_quoted_json_array_is_unwrapped():
    """Python style quoting, which is what a model echoes back from a prompt."""
    assert validate_arguments(ARRAY, {"refs": "['S2']"})["refs"] == ["S2"]


def test_a_comma_separated_string_becomes_a_list():
    assert validate_arguments(ARRAY, {"refs": "S1, S2 ,S3"})["refs"] == ["S1", "S2", "S3"]


def test_a_bare_item_becomes_a_list_of_one():
    assert validate_arguments(ARRAY, {"refs": "S7"})["refs"] == ["S7"]
    assert validate_arguments(ARRAY, {"refs": 3})["refs"] == ["3"]


def test_an_empty_array_string_is_empty():
    assert validate_arguments(spec({"refs": {"type": "array"}}), {"refs": "[]"})["refs"] == []


def test_numbers_arrive_as_strings_and_are_coerced():
    s = spec({"limit": {"type": "integer"}, "score": {"type": "number"}})
    out = validate_arguments(s, {"limit": "5", "score": "0.5"})
    assert out == {"limit": 5, "score": 0.5}


def test_booleans_arrive_in_every_spelling():
    s = spec({"flag": {"type": "boolean"}})
    for truthy in (True, "true", "True", "yes", "1", "on"):
        assert validate_arguments(s, {"flag": truthy})["flag"] is True
    for falsy in (False, "false", "no", "0", "off"):
        assert validate_arguments(s, {"flag": falsy})["flag"] is False


def test_a_boolean_is_not_accepted_as_a_number():
    with pytest.raises(ToolError, match="boolean"):
        validate_arguments(spec({"n": {"type": "integer"}}), {"n": True})


def test_out_of_range_numbers_are_clamped_rather_than_rejected():
    s = spec({"limit": {"type": "integer", "minimum": 1, "maximum": 20}})
    assert validate_arguments(s, {"limit": 500})["limit"] == 20
    assert validate_arguments(s, {"limit": -3})["limit"] == 1


def test_defaults_fill_in_omitted_arguments():
    s = spec({"limit": {"type": "integer", "default": 8}, "q": {"type": "string"}}, ["q"])
    assert validate_arguments(s, {"q": "x"}) == {"limit": 8, "q": "x"}


def test_a_missing_required_argument_is_named_in_the_error():
    with pytest.raises(ToolError, match="needs refs"):
        validate_arguments(ARRAY, {})


def test_an_invented_argument_is_named_in_the_error():
    with pytest.raises(ToolError, match="no argument named"):
        validate_arguments(ARRAY, {"refs": ["S1"], "colour": "blue"})


def test_an_enum_outside_its_values_is_rejected():
    s = spec({"mode": {"type": "string", "enum": ["a", "b"]}})
    with pytest.raises(ToolError, match="must be one of"):
        validate_arguments(s, {"mode": "c"})


def test_an_empty_tool_list_means_no_tools():
    """A role that declares no tools must not be handed the whole toolbox."""
    from mas.kernel.tool import Tool, ToolRegistry

    class Any(Tool):
        name = "any"
        description = "d"
        parameters = {"type": "object", "properties": {}}

        def call(self, ctx, **kw):
            return None

    registry = ToolRegistry()
    registry.register(Any())
    assert registry.available(None, ()) == []
    assert [t.name for t in registry.available(None, ("any",))] == ["any"]

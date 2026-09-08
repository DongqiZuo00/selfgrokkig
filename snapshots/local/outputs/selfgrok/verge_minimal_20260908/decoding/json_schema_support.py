"""Explicit, semantics-preserving expansion of object/array JSON constants.

No proposals are generated here. The caller supplies the frozen catalogue schema.
Unsupported combinations fail instead of silently relaxing the output language.
"""
import copy


def _fixed(value):
    if isinstance(value, dict):
        return {"type": "object", "properties": {k: _fixed(v) for k, v in value.items()},
                "required": list(value), "additionalProperties": False}
    if isinstance(value, list):
        return {"type": "array", "prefixItems": [_fixed(v) for v in value],
                "minItems": len(value), "maxItems": len(value), "items": False}
    return {"const": value}


def expand_structured_constants(schema):
    """Return an expanded schema and the number of explicit structured consts.

    Expansion is declared in worker metadata, never an error-triggered fallback.
    Structural constraints alongside const need a caller-supplied explicit schema
    rather than dropping or incorrectly intersecting those constraints here.
    """
    count = 0

    def visit(value):
        nonlocal count
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return copy.deepcopy(value)
        constant = value.get("const")
        if isinstance(constant, (dict, list)):
            allowed = {"const", "type", "title", "description", "default"}
            if set(value) - allowed:
                raise ValueError("structured const with additional constraints must be expanded explicitly by the caller")
            expected_type = "object" if isinstance(constant, dict) else "array"
            if value.get("type", expected_type) != expected_type:
                raise ValueError("structured const contradicts its type")
            count += 1
            result = _fixed(constant)
            result.update({key: copy.deepcopy(value[key]) for key in ("title", "description", "default") if key in value})
            return result
        # Property names, enums/defaults and other literal data are not themselves
        # schema keywords. Recurse only through actual JSON-schema positions.
        maps = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
        many = {"allOf", "anyOf", "oneOf", "prefixItems"}
        single = {"items", "additionalProperties", "unevaluatedProperties", "additionalItems",
                  "contains", "not", "if", "then", "else", "propertyNames"}
        result = {}
        for key, item in value.items():
            if key in maps:
                result[key] = {name: visit(subschema) for name, subschema in item.items()}
            elif key in many or key in single:
                result[key] = visit(item)
            else:
                result[key] = copy.deepcopy(item)
        return result

    if not isinstance(schema, dict):
        raise ValueError("the caller must provide a JSON-schema object")
    return visit(schema), count

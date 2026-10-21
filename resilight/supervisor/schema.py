import json

from ..modes import MODES

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": MODES},
        "affected_intersections": {
            "anyOf": [{"type": "string", "enum": ["all"]}, {"type": "array", "items": {"type": "string"}}]
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["mode", "affected_intersections", "confidence", "rationale"],
    "additionalProperties": False,
}

GBNF = r'''
root ::= "{" ws "\"mode\"" ws ":" ws mode ws "," ws "\"affected_intersections\"" ws ":" ws scope ws "," ws "\"confidence\"" ws ":" ws num ws "," ws "\"rationale\"" ws ":" ws str ws "}"
mode ::= "\"NORM\"" | "\"INC\"" | "\"SRG\"" | "\"DEG\"" | "\"EMG\""
scope ::= "\"all\"" | "[" ws (str (ws "," ws str)*)? ws "]"
num ::= "0" ("." [0-9]+)? | "1" ("." "0"+)?
str ::= "\"" ([^"\\\n])* "\""
ws ::= [ \t\n]*
'''


def parse(text, valid_ids):
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != set(JSON_SCHEMA["required"]):
        return None
    if obj["mode"] not in MODES:
        return None
    c = obj["confidence"]
    if not isinstance(c, (int, float)) or not 0.0 <= c <= 1.0:
        return None
    scope = obj["affected_intersections"]
    if scope != "all":
        if not isinstance(scope, list) or any(s not in valid_ids for s in scope):
            return None
    if not isinstance(obj["rationale"], str):
        return None
    return obj

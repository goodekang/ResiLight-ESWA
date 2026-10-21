from ..config import CONFIG_DIR
from ..digest import to_json

_DIR = CONFIG_DIR / "prompts"


def build_system_prompt(localization_rule=True, decision_rules=True):
    text = (_DIR / "supervisor_system.txt").read_text(encoding="utf-8")
    loc = (_DIR / "localization_rule.txt").read_text(encoding="utf-8").strip()
    if not decision_rules:
        head, rest = text.split("[Decision rules]")
        _, tail = rest.split("[Network digest", 1)
        return head + "[Network digest" + tail
    return text.replace("{{LOCALIZATION_RULE}}", loc if localization_rule else "Choose between INC and SRG using your judgment of the digest.")


def build_messages(digest, localization_rule=True, decision_rules=True):
    return [
        {"role": "system", "content": build_system_prompt(localization_rule, decision_rules)},
        {"role": "user", "content": to_json(digest)},
    ]

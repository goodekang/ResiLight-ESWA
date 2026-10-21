from dataclasses import dataclass, field

from ..modes import MODES, MODE_INDEX, NORM
from .backend import timed
from .prompt import build_messages
from .schema import parse


@dataclass
class SupervisorDecision:
    mode: int = NORM
    scope: object = "all"
    fallback: bool = False
    raw_mode: int = NORM
    confidence: float = 0.0
    rationale: str = ""
    valid: bool = True
    latency: float = 0.0
    switched: bool = False


@dataclass
class Supervisor:
    backend: object = None
    classifier: object = None
    valid_ids: list = field(default_factory=list)
    confidence_threshold: float = 0.7
    consecutive: int = 2
    budget_s: float = 5.0
    max_failures: int = 3
    localization_rule: bool = True
    decision_rules: bool = True
    validity_check: bool = True
    seed: int = 0

    def __post_init__(self):
        self.mode = NORM
        self.scope = "all"
        self.failures = 0
        self.last_proposal = None
        self.fallback = False
        self.log = []

    def reset(self):
        self.__post_init__()

    def _propose(self, digest):
        if self.classifier is not None:
            return {"mode": MODES[self.classifier.classify(digest)], "affected_intersections": "all",
                    "confidence": 1.0, "rationale": ""}, 0.0
        msgs = build_messages(digest, self.localization_rule, self.decision_rules)
        text, dt = timed(self.backend, msgs, self.budget_s, self.seed)
        return (parse(text, set(self.valid_ids)) if text is not None else None), dt

    def step(self, digest):
        obj, dt = self._propose(digest)
        dec = SupervisorDecision(latency=dt)
        if obj is None and self.validity_check:
            self.failures += 1
            dec.valid = False
            if self.failures >= self.max_failures:
                self.fallback = True
        elif obj is None:
            dec.valid = False
        else:
            self.failures = 0
            self.fallback = False
            proposed = MODE_INDEX[obj["mode"]]
            dec.raw_mode, dec.confidence, dec.rationale = proposed, float(obj["confidence"]), obj["rationale"]
            if not self.validity_check:
                self.mode, self.scope = proposed, obj["affected_intersections"]
                dec.switched = True
            elif obj["confidence"] >= self.confidence_threshold or proposed == self.last_proposal:
                dec.switched = proposed != self.mode
                self.mode, self.scope = proposed, obj["affected_intersections"]
            self.last_proposal = proposed
        dec.mode, dec.scope, dec.fallback = self.mode, self.scope, self.fallback
        self.log.append(dec)
        return dec

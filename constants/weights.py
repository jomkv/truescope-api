VERDICT_FUZZY_MAP = {
    "TRUE": 1.0,
    "MOSTLY-TRUE": 0.85,
    "HALF-TRUE": 0.65,
    "UNPROVEN": 0.5,
    "UNKNOWN": 0.5,
    "MISSING-CONTEXT": 0.45,
    "MISLEADING": 0.3,
    "MOSTLY-FALSE": 0.2,
    "FALSE": 0.0,
    "OUTDATED": 0.5,
    "SATIRE": 0.0,
}

SOURCE_BIAS_WEIGHT = {
    "LEAST-BIASED": 1.0,
    "LEFT-CENTER": 0.9,
    "RIGHT-CENTER": 0.9,
    "RIGHT": 0.8,
    "NEUTRAL": 0.7,
}

NLI_LABEL_SCORE = {
    "entailment": 1.0,
    "neutral": 0.5,
    "contradiction": 0.0,
}

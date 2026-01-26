from constants.enums import Verdict, SourceBias, NLILabel


VERDICT_WEIGHT_MAP: dict[Verdict, float] = {
    Verdict.TRUE: 1.0,
    Verdict.MOSTLY_TRUE: 0.8,
    Verdict.HALF_TRUE: 0.65,
    Verdict.UNPROVEN: 0.5,
    Verdict.UNKNOWN: 0.5,
    Verdict.MISSING_CONTEXT: 0.45,
    Verdict.MISLEADING: -0.3,
    Verdict.MOSTLY_FALSE: -0.8,
    Verdict.FALSE: -1.0,
    Verdict.OUTDATED: 0.5,
    Verdict.SATIRE: 0.0,
}

SOURCE_BIAS_WEIGHT_MAP: dict[SourceBias, float] = {
    SourceBias.LEAST_BIASED: 1.0,
    SourceBias.LEFT_CENTER: 0.9,
    SourceBias.RIGHT_CENTER: 0.9,
    SourceBias.RIGHT: 0.8,
    SourceBias.NEUTRAL: 0.7,
}

NLI_LABEL_WEIGHT_MAP: dict[NLILabel, float] = {
    NLILabel.SUPPORT: 1.0,
    NLILabel.NEUTRAL: 0.5,
    NLILabel.REFUTE: -1.0,
}

from enum import Enum


class VerdictEnum(str, Enum):
    TRUE = "TRUE"
    MOSTLY_TRUE = "MOSTLY_TRUE"
    HALF_TRUE = "HALF_TRUE"
    MOSTLY_FALSE = "MOSTLY_FALSE"
    FALSE = "FALSE"
    UNPROVEN = "UNPROVEN"
    MISLEADING = "MISLEADING"
    SATIRE = "SATIRE"
    MISSING_CONTEXT = "MISSING_CONTEXT"
    OUTDATED = "OUTDATED"
    UNKNOWN = "UNKNOWN"


VERDICT_SCORE_MAP: dict[VerdictEnum, float] = {
    VerdictEnum.TRUE: 1.0,
    VerdictEnum.MOSTLY_TRUE: 0.8,
    VerdictEnum.HALF_TRUE: 0.5,
    VerdictEnum.MOSTLY_FALSE: 0.2,
    VerdictEnum.FALSE: 0.0,
    VerdictEnum.UNPROVEN: 0.4,
    VerdictEnum.MISLEADING: 0.2,
    VerdictEnum.SATIRE: 0.1,
    VerdictEnum.MISSING_CONTEXT: 0.5,
    VerdictEnum.OUTDATED: 0.3,
    VerdictEnum.UNKNOWN: 0.5,
}

from enum import Enum


class Verdict(str, Enum):
    TRUE = "TRUE"
    MOSTLY_TRUE = "MOSTLY-TRUE"
    HALF_TRUE = "HALF-TRUE"
    UNPROVEN = "UNPROVEN"
    UNKNOWN = "UNKNOWN"
    MISSING_CONTEXT = "MISSING-CONTEXT"
    MISLEADING = "MISLEADING"
    MOSTLY_FALSE = "MOSTLY-FALSE"
    FALSE = "FALSE"
    OUTDATED = "OUTDATED"
    SATIRE = "SATIRE"


class SourceBias(str, Enum):
    LEFT = "LEFT"
    LEFT_CENTER = "LEFT-CENTER"
    LEAST_BIASED = "LEAST-BIASED"
    RIGHT_CENTER = "RIGHT-CENTER"
    RIGHT = "RIGHT"
    NEUTRAL = "NEUTRAL"  # For sources with unknown source bias


class NLILabel(str, Enum):
    SUPPORT = "support"
    NEUTRAL = "neutral"
    REFUTE = "refute"


class StreamEventType(str, Enum):
    SEARCH_HITS = "search_hits"
    RESULT = "result"
    STATS = "stats"
    REMARKS = "remarks"
    COMPLETE = "complete"
    ERROR = "error"


# Generic entity terms that should not dominate partial entity matching
ENTITY_GENERIC_TOKENS = {
    # Weather descriptors
    "super",
    "typhoon",
    "tropical",
    "storm",
    "depression",
    "bagyo",
    "bagyong",
    # Political/titles
    "president",
    "vice",
    "senator",
    "mayor",
    "governor",
    "congressman",
    "minister",
    "secretary",
    "representative",
    # Business/organization descriptors
    "company",
    "corporation",
    "inc",
    "corp",
    "ltd",
    "llc",
    "foundation",
    "organization",
    # Geographic descriptors
    "city",
    "province",
    "municipality",
    "region",
    "district",
    "barangay",
    "island",
    "country",
    # General descriptors
    "new",
    "old",
    "national",
    "international",
    "global",
    "local",
    "federal",
    "state",
    "central",
    "general",
}

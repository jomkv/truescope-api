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
    "sea",
    "ocean",
    "river",
    "lake",
    "coast",
    "shores",
    "west",
    "east",
    "north",
    "south",
    "central",
    "northern",
    "southern",
    "eastern",
    "western",
    "territory",
    # Military/Maritime
    "guard",
    "vessel",
    "ship",
    "boat",
    "military",
    "forces",
    "force",
    "navy",
    "army",
    "air",
    "base",
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
    # Topical/Numerical descriptors
    "percent", "percentage", "rate", "rates", "level", "levels",
    "amount", "value", "total", "average", "number", "numbers",
    "year", "years", "data", "report", "claims", "claim",
}

# Common English stopwords/noise words to exclude from topical relevance points
COMMON_STOPWORDS = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "the", "a", "an", "this", "that", "these", "those",
    "and", "but", "or", "not", "for", "with", "from", "at", "by", "of", "to", "in", "on",
    "it", "he", "she", "they", "we", "you", "i", "my", "his", "her", "their", "our",
    "who", "which", "what", "can", "will", "would", "should", "all", "any", "some", "most",
    "percent", "percentage", "rate", "rates",
    "against", "under", "over", "after", "before", "around", "between", "through",
    "say", "says", "said", "claiming", "claim", "claims", "report", "reported",
}

EVENT_MARKERS = {
    "super", "typhoon", "tropical", "storm", "depression", "bagyo", "bagyong",
    "city", "province", "municipality", "region", "district", "barangay", "island", "country",
    "sea", "ocean", "river", "lake", "coast", "shores",
    "vessel", "ship", "boat", "navy", "guard"
}

STOP_TITLES = {"president", "vice", "senator", "inc", "corp"}

# Keywords for dampening scores of video/social media debunks
DAMP_KEYWORDS = {"video", "post", "social media", "tiktok", "facebook", "ai-generated", "manipulated"}
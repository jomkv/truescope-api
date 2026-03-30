import spacy
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
    "democratic",
    "senate",
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
    "york",
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
    "year", "years", "data", "report", "claims", "claim", "car",
    "trade"
}

# Initialize a blank English model to access default stopwords
# This is fast as it doesn't load any model weights
nlp = spacy.blank("en")

# Common English stopwords/noise words to exclude from topical relevance points
# We combine spaCy's defaults with domain-specific terms (metrics, reporting verbs)
COMMON_STOPWORDS = nlp.Defaults.stop_words.union({
    "percent", "percentage", "rate", "rates", "level", "levels",
    "amount", "value", "total", "average", "number", "numbers",
    "claiming", "claim", "claims", "report", "reported",
    "say", "says", "said"
})

EVENT_MARKERS = {
    "super", "typhoon", "tropical", "storm", "depression", "bagyo", "bagyong",
    "city", "province", "municipality", "region", "district", "barangay", "island", "country",
    "sea", "ocean", "river", "lake", "coast", "shores",
    "vessel", "ship", "boat", "navy", "guard"
}

STOP_TITLES = {"president", "vice", "senator", "inc", "corp"}

# Keywords for dampening scores of video/social media debunks
DAMP_KEYWORDS = {"video", "post", "social media", "tiktok", "facebook", "ai-generated", "manipulated"}

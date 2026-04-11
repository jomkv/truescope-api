from schemas.article_schema import Article
from schemas.claim_schema import Claim
from schemas.article_chunk_schema import ArticleChunk
from models.article_result_model import ArticleResultModel
from constants.weights import (
    VERDICT_WEIGHT_MAP,
    SOURCE_BIAS_WEIGHT_MAP,
    NLI_LABEL_WEIGHT_MAP,
)
from constants.enums import (
    Verdict,
    NLILabel,
    SourceBias,
    StreamEventType,
)
from constants.fuzzy import (
    DEMONYM_GROUPS,
    COMMON_PLURAL_SUFFIXES,
    ANTONYM_PAIRS,
    MIN_STEM_MATCH_LENGTH,
)
from constants.tokens import (
    ENTITY_GENERIC_TOKENS,
    COMMON_STOPWORDS,
    EVENT_MARKERS,
    STOP_TITLES,
    DAMP_KEYWORDS,
)
from constants.negations import (
    NEGATION_PHRASES,
    NEGATION_WORD_PATTERNS,
    NEGATION_TOKENS,
)
from dateparser.search import search_dates
from services import (
    EmbeddingService,
    EntityExtractionService,
    NLIService,
    RemarksGenerationService,
    StatsService,
)
import numpy as np
import unicodedata
import re
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import cast
from collections import defaultdict
from databases.verify import VerifyDatabase


class VerifyController:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.entity_extraction_service = EntityExtractionService()
        self.nli_service = NLIService()
        self.remarks_generation_service = RemarksGenerationService()
        self.stats_service = StatsService()
        self.db = VerifyDatabase()

        # Balanced thresholds — fine-tuned embeddings via HITL improve precision,
        # but we keep gates slightly relaxed to catch all genuinely related evidence.
        self.RELEVANCE_THRESHOLD = 0.24
        self.ENTITY_THRESHOLD = 0.25
        self.COMBINED_THRESHOLD = 0.25

        # Weights for combined relevance score (70% semantic, 30% entity)
        self.SEMANTIC_WEIGHT = 0.7
        self.ENTITY_WEIGHT = 0.3

        # Thread pool for CPU-intensive ML operations (2 workers optimal for PyTorch models)
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Scale analysis limits to provide more comprehensive results (Territorial/Complex claims)
        self.AGGREGATION_LIMIT = 3
        self.DB_RETRIEVE_LIMIT = 20
        self.NLI_CONFIDENCE_GATE = 0.45
        self.UNCERTAINTY_THRESHOLD = 0.85

    @staticmethod
    def normalize_text(
        text: str, lowercase: bool = True, strip_punctuation: bool = False
    ) -> str:
        """
        Normalize input text for consistent downstream processing.

        Steps:
        - Unicode normalization (NFKC)
        - Optional punctuation stripping (helps tokenization boundaries)
        - Optional lowercasing
        - Removal of extra whitespace
        - Stripping leading/trailing whitespace

        Args:
            text (str): The input text to normalize.
            lowercase (bool): Whether to lowercase the text (default True).
            strip_punctuation (bool): Whether to replace special chars with spaces (default False).

        Returns:
            str: The normalized text.
        """
        if not isinstance(text, str):
            return ""

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Optional punctuation stripping replace with space to preserve token boundaries
        if strip_punctuation:
            text = re.sub(r"[^\w\s-]", " ", text)

        # Lowercase if requested
        if lowercase:
            text = text.lower()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        return text.strip()

    def get_chunk_map(
        self,
        embedding: list[float],
        doc_ids: set[str],
        top_n: int = 3,
    ) -> dict[str, list[tuple[ArticleChunk, float]]]:
        """
        For each doc_id in doc_ids, finds the most relevant article chunks.
        Optimization: Uses batch non-vector retrieval followed by local numpy ranking.
        """
        # Fetch all chunks (now including embeddings)
        all_doc_chunks = self.db.find_chunks_by_doc_ids(doc_ids)

        # Prepare query vector
        query_vec = np.array(embedding)
        query_norm = np.linalg.norm(query_vec)

        # Group and rank chunks by doc_id
        chunk_map: dict[str, list[tuple[ArticleChunk, float]]] = defaultdict(list)
        for chunk in all_doc_chunks:
            similarity = 0.0
            if chunk.embedding:
                chunk_vec = np.array(chunk.embedding)
                chunk_norm = np.linalg.norm(chunk_vec)
                if query_norm > 0 and chunk_norm > 0:
                    similarity = float(
                        np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm)
                    )

            chunk_map[chunk.doc_id].append((chunk, similarity))

        # Sort each doc_id's chunks by similarity descending and take top_n
        for doc_id in chunk_map:
            chunk_list = chunk_map[doc_id]
            chunk_list.sort(key=lambda x: x[1], reverse=True)
            chunk_map[doc_id] = chunk_list[:top_n]

        return chunk_map

    def get_article_map(self, doc_ids: set[str]) -> dict[str, Article]:
        """
        Retrieves articles from the database for the given set of doc_ids.

        Args:
            doc_ids (set[str]): Set of unique document IDs.

        Returns:
            dict[str, Article]: Mapping from doc_id to Article object.
        """
        article_results = self.db.find_articles_from_doc_ids(doc_ids)
        article_map: dict[str, Article] = {
            article.doc_id: article for article in article_results
        }

        return article_map

    @staticmethod
    def extract_unique_doc_ids(article_chunks: list[ArticleChunk | Claim]) -> set[str]:
        """
        Extracts unique document IDs from a list of ArticleChunk or Claim objects.

        Args:
            article_chunks (list[ArticleChunk | Claim]): List of article chunks or claims.

        Returns:
            set[str]: Set of unique document IDs.
        """
        unique_doc_ids: set[str] = set()

        for chunk, _ in article_chunks:
            doc_id = chunk.doc_id
            if doc_id not in unique_doc_ids:
                unique_doc_ids.add(doc_id)

        return unique_doc_ids

    def find_claims_with_articles(
        self, embedding: list[float], top_k: int = 20, exclude_doc_ids: list[str] = None
    ) -> list[tuple[Claim, Article, float, str | None]]:
        """
        Retrieves the top_k most similar claims from the database using HNSW vector search,
        skipping claims with an 'UNKNOWN' verdict, and pairs each claim with its corresponding article.

        Args:
            embedding (list[float]): Embedding vector for the user claim.
            top_k (int, optional): Number of top similar claims to retrieve. Defaults to 20.
            exclude_doc_ids (list[str], optional): List of document IDs to exclude from the search. Defaults to None.

        Returns:
            list[tuple[Claim, Article, float, str | None]]: List of tuples containing the claim, article, similarity score, and relevant chunk text.
        """
        similar_claims = self.db.find_similar_claims(embedding, top_k)
        unique_doc_ids = self.extract_unique_doc_ids(similar_claims)

        # Get equivalent articles for all found vectors
        article_map = self.get_article_map(unique_doc_ids)

        # Batch-load all chunks at once (N+1 query fix)
        all_chunks_map = self.get_chunk_map(embedding, unique_doc_ids)

        # Combine all results into a tuple
        # (claim, article, similarity_score, relevant_chunk_text)
        results: list[tuple[Claim, Article, float, str | None]] = []

        for claim, distance in similar_claims:
            article = cast(Article, article_map.get(claim.doc_id))
            if exclude_doc_ids and claim.doc_id in exclude_doc_ids:
                continue
            article_relevant_chunks = all_chunks_map[article.doc_id]

            if len(article_relevant_chunks) == 0:
                results.append((claim, article, 1 - distance, None))
                continue

            chunk_text = self.build_chunk_text(
                [chunk for chunk, _ in article_relevant_chunks]
            )
            results.append((claim, article, 1 - distance, chunk_text))

        return results

    def find_news_articles(
        self, embedding: list[float], top_k: int = 20
    ) -> list[tuple[Article, float, str | None]]:
        """
        Retrieves the top_k most relevant news articles, grouping and ranking based on
        chunk similarity. Powered by high-speed Light HNSW index.

        Args:
            embedding (list[float]): Embedding vector for the user claim.
            top_k (int, optional): Number of top similar articles to retrieve. Defaults to 20.

        Returns:
            list[tuple[Article, float, str | None]]: List of tuples containing the article, similarity score, and relevant chunk text.
        """
        # 1: Get similar chunks via DiskANN index (Fast & Global)
        chunk_results = self.db.find_similar_chunks(embedding, top_k)

        # 2: Collect unique doc_ids from chunks
        unique_doc_ids = self.extract_unique_doc_ids(chunk_results)

        # 3: Get Equivalent Articles for all found vectors
        article_map = self.get_article_map(unique_doc_ids)

        # 4: Batch-load all chunks for these specific articles
        all_chunks_map = self.get_chunk_map(embedding, unique_doc_ids)

        # 5: Combine into final tuples
        results: list[tuple[Article, float, str | None]] = []
        for chunk, distance in chunk_results:
            article = cast(Article, article_map.get(chunk.doc_id))
            if not article:
                continue

            article_relevant_chunks = all_chunks_map.get(article.doc_id, [])
            chunk_text = (
                self.build_chunk_text([c for c, _ in article_relevant_chunks])
                if article_relevant_chunks
                else None
            )

            results.append((article, 1 - distance, chunk_text))

        return results

    def extract_entities(self, text: str) -> list[str]:
        """
        Extracts named entities from the given text using the entity extraction service.

        Args:
            text (str): The input text from which to extract entities.

        Returns:
            list[str]: List of extracted entity names.
        """
        entities_with_label = self.entity_extraction_service.extract_entities(text)

        # Extract entity names only, exclude label
        entities = [entity[0] for entity in entities_with_label if entity and entity[0]]

        # De-duplicate while preserving order (case-insensitive)
        deduped_entities: list[str] = []
        seen_entities: set[str] = set()
        for entity in entities:
            norm_entity = self.normalize_text(entity)
            if not norm_entity or norm_entity in seen_entities:
                continue
            seen_entities.add(norm_entity)
            deduped_entities.append(entity)

        return deduped_entities

    @staticmethod
    def tokenize_text(text: str) -> set[str]:
        """
        Tokenize input text into a set of words.
        Uses a robust regex to find letter-based sequences (Unicode inclusive).
        """
        return set(re.findall(r"[^\W\d_]{2,}", text))

    @staticmethod
    def is_fuzzy_match(t1: str, t2: str) -> bool:
        """
        Check if two tokens are essentially the same (plural, stem variant, demonym).
        Uses generalized rules and patterns from constants.fuzzy.
        """

        if t1 == t2:
            return True

        # 1. Case-insensitive normalization
        t1_l, t2_l = t1.lower(), t2.lower()

        # 2. Demonym/Related term groups (Checked first to allow short terms like "us" or "uk")
        for group in DEMONYM_GROUPS:
            if t1_l in group and t2_l in group:
                return True

        if len(t1) < 3 or len(t2) < 3:
            return False

        # 3. Common plurals/suffixes
        for suffix in COMMON_PLURAL_SUFFIXES:
            if t1 + suffix == t2 or t2 + suffix == t1:
                return True

            # Handle -y to -ies (e.g. city -> cities)
            if suffix == "ies":
                if (t1.endswith("y") and t1[:-1] + "ies" == t2) or (
                    t2.endswith("y") and t2[:-1] + "ies" == t1
                ):
                    return True

        # 4. Stem variants (e.g. impeached/impeachment)
        # Shared prefix of at least MIN_STEM_MATCH_LENGTH chars or long/short variant (e.g. Philippine/Philippines)
        s, l = (t1_l, t2_l) if len(t1_l) < len(t2_l) else (t2_l, t1_l)
        if len(s) >= 4 and l.startswith(s):
            return True

        # Check for shared prefix (e.g. "impeach" matches "impeached" and "impeachment")
        if (
            len(t1_l) >= MIN_STEM_MATCH_LENGTH
            and len(t2_l) >= MIN_STEM_MATCH_LENGTH
            and t1_l[:MIN_STEM_MATCH_LENGTH] == t2_l[:MIN_STEM_MATCH_LENGTH]
        ):
            return True

        return False

    def calculate_entity_match_score(
        self, claim_entities: list[str], text: str, article_title: str = ""
    ) -> float:
        """
        Calculates the entity match score between a list of claim entities and the provided text/article title.
        The score represents the proportion of claim entities found as whole words in the text or title.

        Args:
            claim_entities (list[str]): List of entities extracted from the claim.
            text (str): The text (e.g., article content) to search for entities.
            article_title (str, optional): The article title to also search for entities. Defaults to "".

        Returns:
            float: Entity match score between 0.0 and 1.0.
        """
        if not claim_entities:
            return 0.0

        text_norm = self.normalize_text(text, strip_punctuation=True)
        article_title_norm = self.normalize_text(article_title, strip_punctuation=True)

        matches = 0.0
        total_weight = 0.0

        # Tokenize the normalized article content and title into sets of words
        text_tokens = self.tokenize_text(text_norm)
        title_tokens = self.tokenize_text(article_title_norm)
        # Combine tokens from both content and title for comparison
        comparison_tokens = text_tokens.union(title_tokens)

        for entity in claim_entities:
            # Normalize the entity string with stripping to match article boundaries
            entity_lower = self.normalize_text(entity, strip_punctuation=True)
            # Tokenize the entity into words
            entity_tokens_all = self.tokenize_text(entity_lower)

            # Identify specific (non-generic) tokens in the entity
            specific_entity_tokens = [
                token
                for token in entity_tokens_all
                if token not in ENTITY_GENERIC_TOKENS and len(token) >= 3
            ]

            # Assign lower weight to generic-only entities, full weight to those with specifics
            entity_weight = (
                0.25 if entity_tokens_all and not specific_entity_tokens else 1.0
            )
            total_weight += entity_weight

            # Match as whole word in text or title
            if re.search(
                r"\b" + re.escape(entity_lower) + r"\b", text_norm
            ) or re.search(r"\b" + re.escape(entity_lower) + r"\b", article_title_norm):
                matches += entity_weight
                continue

            # Partial token match fallback for incomplete mentions
            if specific_entity_tokens:
                found_partial_count = 0
                for set_token in specific_entity_tokens:
                    if set_token in comparison_tokens:
                        found_partial_count += 1
                        continue
                    for comp_token in comparison_tokens:
                        if self.is_fuzzy_match(set_token, comp_token):
                            found_partial_count += 1
                            break

                if found_partial_count > 0:
                    # Give partial credit based on how many tokens matched
                    coverage = found_partial_count / len(specific_entity_tokens)
                    matches += coverage * 0.9 * entity_weight

        return matches / total_weight if total_weight > 0 else 0.0

    def requires_specific_entity_match(self, claim_entities: list[str]) -> bool:
        """
        Returns True when claim entities contain both generic descriptors and specific named tokens.
        Examples:
        - "President Marcos" -> True (has generic "president" + specific "marcos")
        - "Super Typhoon Uwan" -> True (has generic "super"/"typhoon" + specific "uwan")
        - "Apple Inc" -> True (has generic "inc" + specific "apple")
        - "John Smith" -> False (no generic descriptors, just specific names)
        - "The President" -> False (only generic, no specific name)
        """
        has_generic_descriptor = False
        has_specific_token = False

        for entity in claim_entities:
            norm_entity = self.normalize_text(entity)
            entity_tokens = list(self.tokenize_text(norm_entity))
            for token in entity_tokens:
                if token in ENTITY_GENERIC_TOKENS:
                    has_generic_descriptor = True
                elif len(token) >= 3:
                    has_specific_token = True

        return has_generic_descriptor and has_specific_token

    def has_specific_entity_token_match(
        self, claim_entities: list[str], text: str, article_title: str = ""
    ) -> bool:
        """
        Checks whether at least one specific (non-generic) claim entity token appears
        in the comparison text/title.
        """
        # Always use robust stripping for matching tokens
        text_norm = self.normalize_text(text, strip_punctuation=True)
        article_title_norm = self.normalize_text(article_title, strip_punctuation=True)
        comparison_tokens = self.tokenize_text(text_norm)
        comparison_tokens.update(self.tokenize_text(article_title_norm))

        for entity in claim_entities:
            entity_tokens = self.tokenize_text(
                self.normalize_text(entity, strip_punctuation=True)
            )
            specific_tokens = [
                token
                for token in entity_tokens
                if token not in ENTITY_GENERIC_TOKENS and len(token) >= 3
            ]
            if any(
                any(self.is_fuzzy_match(st, ct) for ct in comparison_tokens)
                for st in specific_tokens
            ):
                return True

        return False

    @staticmethod
    def build_chunk_text(
        chunks: list[ArticleChunk],
        max_chars: int = 800,
        max_chunks: int = 3,
    ) -> str | None:
        """
        Builds a short context string for remarks generation by concatenating up to max_chunks chunk excerpts,
        with a total character limit of max_chars.

        Args:
            chunks (list[ArticleChunk]): List of article chunks.
            max_chars (int, optional): Maximum total characters in the output. Defaults to 800.
            max_chunks (int, optional): Maximum number of chunks to include. Defaults to 3.

        Returns:
            str | None: Concatenated chunk text or None if no chunks are provided.
        """
        if len(chunks) == 0:
            return None

        parts: list[str] = []
        used_chars = 0

        for chunk in chunks:
            if len(parts) >= max_chunks or used_chars >= max_chars:
                break

            remaining = max_chars - used_chars
            content = chunk.chunk_content.strip()
            if not content:
                continue

            snippet = content[:remaining].rsplit(" ", 1)[0].strip()
            if not snippet:
                continue

            parts.append(snippet)
            used_chars += len(snippet) + 1

        return "\n\n".join(parts).strip()

    @staticmethod
    def truncate_at_sentence(text: str, max_chars: int = 200) -> str:
        """
        Truncates the input text at the nearest sentence boundary within max_chars characters.

        Args:
            text (str): The input text to truncate.
            max_chars (int, optional): Maximum number of characters. Defaults to 200.

        Returns:
            str: Truncated text ending at a sentence boundary or word.
        """
        cleaned = re.sub(r"\s+", " ", text.strip())
        if len(cleaned) <= max_chars:
            return cleaned

        truncated = cleaned[:max_chars]
        sentence_matches = list(re.finditer(r"[.!?](?=\s+[A-Z0-9]|$)", truncated))

        if sentence_matches:
            last_end = sentence_matches[-1].end()
            return cleaned[:last_end].strip()

        return truncated.rsplit(" ", 1)[0].rstrip() + "..."

    def extract_claim_timeframe(self, user_claim: str) -> list[tuple[str, datetime]]:
        """
        Extracts temporal references (e.g., dates, periods) from a claim using dateparser.

        Args:
            claim (str): The user-provided claim to analyze for time expressions.

        Returns:
            list[tuple[str, datetime]]: List of tuples containing the matched time string and its parsed datetime.
        """

        # TODO: Still unsure on how to use extracted timeframe on the current logic

        return search_dates(user_claim, settings={"RETURN_TIME_SPAN": True})

    def detect_claim_stance(self, claim: str) -> tuple[str, bool]:
        """
        Detects whether a user claim contains an explicit negation/false-stance marker,
        and if so, strips the marker to return the core factual assertion.

        This enables the double-negative scoring logic:
            negated user claim  +  FALSE-verdict DB article  +  NLI SUPPORT
            = negative score (user supports a false claim)
            → flip sign because user claim was negated
            = POSITIVE score → correctly classified as TRUE

        Examples:
            "It is false that Marcos was caught on video using cocaine"
            → core: "Marcos was caught on video using cocaine", is_negated=True

            "Marcos was inaugurated as president"
            → core: "Marcos was inaugurated as president", is_negated=False

        Args:
            claim (str): The original user-provided claim text.

        Returns:
            tuple[str, bool]: (core_claim, is_negated)
                - core_claim: The claim with negation markers stripped (or original if none).
                - is_negated: True if a negation pattern was found and stripped.
        """
        # Negation patterns moved to constants/negations.py

        claim_lower = claim.lower().strip()

        # 1. Check for leading negation PHRASES first (strip and return)
        for phrase in NEGATION_PHRASES:
            if claim_lower.startswith(phrase):
                core = claim[len(phrase) :].strip()
                # Capitalize first letter
                if core:
                    core = core[0].upper() + core[1:]
                return core, True

        # 2. Check for inline negation WORDS/PATTERNS
        for pattern, _ in NEGATION_WORD_PATTERNS:
            match = re.search(pattern, claim, flags=re.IGNORECASE)
            if match:
                # We strip the negation using regex substitution.
                # If the pattern has an auxiliary verb group (e.g. "was not" -> "was"),
                # we use \1 to keep the verb. Otherwise we just strip the whole pattern.
                if match.groups():
                    core = re.sub(pattern, r"\1", claim, flags=re.IGNORECASE)
                else:
                    core = re.sub(pattern, "", claim, flags=re.IGNORECASE)

                # Cleanup: remove double spaces and trim
                core = re.sub(r"\s+", " ", core).strip()

                # Capitalize first letter
                if core:
                    core = core[0].upper() + core[1:]

                return core, True

        # No negation found
        return claim, False

    @staticmethod
    def is_polarity_mismatch(tokens1: set[str], tokens2: set[str]) -> bool:
        """
        Detects if two sets of tokens contain contradictory directional signals.
        Returns True if a contradiction is found (e.g., "win" vs "won't win").
        """
        t1_low = {t.lower() for t in tokens1}
        t2_low = {t.lower() for t in tokens2}

        # 1. Check for Antonym Pairs (Win/Lose)
        antonym_match = False
        for a, b in ANTONYM_PAIRS:
            if (a in t1_low and b in t2_low) or (b in t1_low and a in t2_low):
                antonym_match = True
                break

        # 2. Check for Negation Mismatch
        has_neg1 = any(n in t1_low for n in NEGATION_TOKENS)
        has_neg2 = any(n in t2_low for n in NEGATION_TOKENS)

        negation_match = has_neg1 != has_neg2

        # Logic:
        # - (Negation) + (No Antonym) = Mismatch (e.g., "win" vs "not win")
        # - (No Negation) + (Antonym) = Mismatch (e.g., "win" vs "lose")
        # - (Negation) + (Antonym) = MATCH (e.g., "not win" == "lose") -> Returns False
        if negation_match and antonym_match:
            return False  # They cancel each other out

        return negation_match or antonym_match

    def compute_final_score(
        self,
        verdict: Verdict | None,
        source_bias: SourceBias | None,
        nli_label: NLILabel,
        nli_score: float,
        is_factcheck: bool = True,
        similarity_score: float = 0.0,
        article_content: str = "",
        has_topical_match: bool = False,
        is_negated: bool = False,
    ) -> float:
        """
        Computes a final score for a claim-article pair based on the verdict, source bias, NLI label, and NLI confidence.

        The output is a fuzzified score in the range [-1, 1]:
            - A score close to 1 means strong support for the user claim (completely true).
            - A score close to -1 means strong refutation of the user claim (completely false).
            - A score near 0 means the evidence is neutral or inconclusive.
        The magnitude reflects the strength of the evidence, and the sign reflects the direction (support or refute).

        Args:
            verdict (Verdict | None): The verdict of the found claim (as an enum).
            source_bias (SourceBias | None): The bias of the article's source (as an enum).
            nli_label (NLILabel): The NLI relationship label between user and found claim.
            nli_score (float): The NLI model's confidence score for the label (0.0 to 1.0).
            is_factcheck (bool, optional): Whether the claim is from a fact-checking source. Defaults to True.
            similarity_score (float, optional): Similarity score for news articles. Defaults to 0.0.
            article_content (str, optional): Full article content for context analysis. Defaults to "".
            has_topical_match (bool, optional): Whether the article matches topical assertions of the claim.

        Returns:
            float: The computed final score (signed and fuzzified, with magnitude reflecting strength).
        """
        if not is_factcheck:

            if nli_label == NLILabel.SUPPORT:
                base_score = 0.75
            elif nli_label == NLILabel.REFUTE:
                base_score = -0.75
            elif nli_label == NLILabel.NEUTRAL:
                base_score = 0.0
            else:
                base_score = 0.0

            confidence_multiplier = 0.5 + (nli_score * 0.5)

            bias_weight = (
                SOURCE_BIAS_WEIGHT_MAP.get(source_bias, 0.7) if source_bias else 0.7
            )

            return round(base_score * confidence_multiplier * bias_weight, 2)

        if verdict is None or source_bias is None:
            return 0.0

        verdict_weight = VERDICT_WEIGHT_MAP.get(verdict, 0.5)
        bias_weight = SOURCE_BIAS_WEIGHT_MAP.get(source_bias, 0.7)

        # --- Decision Dampening Logic ---
        # A low-confidence NLI result is worse than no result.
        # We implementation a progressive dampening phase to avoid "flips" from weak signals.
        # REGRESSION FIX: Only apply the progressive dampening (0.65-0.75) if the claim
        # is negated, as negated claims are more sensitive to NLI noise.
        # Standard claims only get dampened if the NLI score is critically low (< 0.55).
        confidence_factor = nli_score
        if nli_score < 0.55:
            # Below the base gate: absolute uncertainty (noise)
            confidence_factor = nli_score * 0.4
        elif is_negated:
            if nli_score < 0.65:
                # Borderline: dampen impact by 30% to avoid swinging the total verdict
                confidence_factor = nli_score * 0.7
            elif nli_score < 0.75:
                # Moderate: dampen slightly (10%)
                confidence_factor = nli_score * 0.9

        if nli_label == NLILabel.REFUTE:
            # For REFUTE, the direction is usually opposite the truth.
            # However, if the Article Verdict is already Negative (FALSE/MISLEADING),
            # then a REFUTE from that source means the User claim is definitely Wrong.

            nli_label_weight = NLI_LABEL_WEIGHT_MAP.get(nli_label, -1.0)

            if verdict_weight < 0:
                # --- FACT-CHECK REFUTATION GUARD ---
                if has_topical_match:
                    # FactCheck specifically debunks user's claim. Force FALSE verdict.
                    nli_label_weight = 1.0

                    # --- DOUBLE-NEGATIVE GUARD ---
                    # If claim is negated and evidence confirms positive false,
                    # NLI REFUTE + FALSE verdict = positive support for negated claim.
                    if is_negated:
                        nli_label_weight = -1.0
                else:
                    nli_label_weight = -1.0

            return round(
                confidence_factor * bias_weight * verdict_weight * nli_label_weight, 2
            )

        # For SUPPORT and NEUTRAL the signed verdict_weight gives the correct direction:
        #   SUPPORT + TRUE  → positive (user claim is true)
        #   SUPPORT + FALSE → negative (user claim matches a debunked claim → false)
        #   NEUTRAL + TRUE  → weakly positive
        #   NEUTRAL + FALSE → weakly negative
        nli_label_weight = NLI_LABEL_WEIGHT_MAP.get(nli_label, 0.5)

        # --- NEGATED CLAIM SUPPORT GUARD ---
        # If user's claim is negated and NLI finds SUPPORT, flip NLI weight.
        # This correctly penalizes negating true facts, and rewards negating false facts.
        if is_negated and nli_label == NLILabel.SUPPORT:
            nli_label_weight = -nli_label_weight

        raw_score = confidence_factor * bias_weight * verdict_weight * nli_label_weight

        # --- AI/Video/Social Media Dampening ---
        # Fact-checks that specifically debunk "videos", "posts", or "AI" content
        # should be dampened if NLI confidence isn't absolute (>= 0.95).
        # This prevents specific video debunks from drowning out broader news reporting.
        if is_factcheck and nli_score < 0.95:
            content_lower = article_content.lower()
            if any(k in content_lower for k in DAMP_KEYWORDS):
                raw_score *= 0.7  # reduce impact by 30%

        return round(raw_score, 2)

    async def verify_claim(
        self,
        user_claim: str,
        use_fallback: bool = True,
        exclude_doc_ids: list[str] = None,
        exclude_articles: bool = False,
        aggregation_limit: int | None = None,
    ):
        """
        Main entry point for verifying a user claim (async version).

        Steps:
        - Embeds the user claim and searches for similar claims in the database.
        - Extracts entities from the user claim and calculates entity match scores.
        - For each found claim-article pair, runs NLI to determine the relationship and computes a final score.
        - Filters and sorts results based on relevance and entity match.

        Args:
            user_claim (str): The user-provided claim to verify.
            use_fallback (bool, optional): Whether to use fallback logic if no strong matches are found. Defaults to True.
            exclude_doc_ids (list[str], optional): List of document IDs to exclude from the search. Defaults to None.

        Returns:
            dict: Dict containing lists of results, each containing article and claim details, NLI results, scores, and skip reasons.
        """

        user_claim_for_matching = self.normalize_text(user_claim, lowercase=False)

        # --- Stance Detection ---
        # Detect negation in the user claim and extract the core assertion.
        # The core claim is used for embedding/matching so the DB finds related articles
        # (e.g., the debunked FALSE article about the same topic).
        # We also use the CORE claim for NLI evaluation to ensure logical consistency
        # when we flip the final sign if is_negated is True.
        core_claim_text, is_negated = self.detect_claim_stance(user_claim_for_matching)
        user_claim_core_norm = self.normalize_text(core_claim_text)

        # Run embedding and entity extraction in parallel
        loop = asyncio.get_event_loop()
        claim_embedding, claim_entities = await asyncio.gather(
            loop.run_in_executor(
                self.executor,
                self.embedding_service.embed_text,
                core_claim_text,  # Use core claim (negation stripped) for embedding
            ),
            loop.run_in_executor(
                self.executor, self.extract_entities, user_claim_for_matching
            ),
        )

        # Phase 1: High-Precision Claim/Fact-Check Search (Truth Search)
        # Search the 48k rows Truth database first.
        similar_claims = self.db.find_similar_claims(claim_embedding, limit=20)

        # Phase 2: Full News Article Search (Zero-waste)
        # Using the new native chunks_vec_idx, we can search the entire 350,000 row table
        # with zero row-read overhead. This ensures we never miss articles.
        similar_chunks = self.db.find_similar_chunks(claim_embedding, limit=20)

        # Phase 3: Hydrate Results for Evaluation Engine (Prevents Unpacking Crash)
        factcheck_results = []
        for claim, distance in similar_claims:
            if (
                exclude_doc_ids
                and claim.article
                and claim.article.doc_id in exclude_doc_ids
            ):
                continue

            # The database returns vector_distance (0=identical, 1=orthogonal).
            # We MUST convert this back to cosine similarity before the semantic weighting pipeline
            # or else the lowest quality matches get the highest exponential rewards.
            similarity_score = max(0.0, 1.0 - float(distance))
            factcheck_results.append(
                (
                    claim,
                    claim.article,
                    similarity_score,
                    (
                        "\n".join(
                            [chunk.chunk_content for chunk in claim.article.chunks]
                        )
                        if claim.article and claim.article.chunks
                        else ""
                    ),
                )
            )

        if exclude_articles:
            news_results = []
        else:
            news_results = []
            for chunk, distance in similar_chunks:
                if (
                    exclude_doc_ids
                    and chunk.article
                    and chunk.article.doc_id in exclude_doc_ids
                ):
                    continue

                similarity_score = max(0.0, 1.0 - float(distance))
                news_results.append(
                    (chunk.article, similarity_score, chunk.chunk_content)
                )

        # Process all results
        processed_doc_ids = set()
        tasks = []

        # --- Evaluation Pipeline ---
        # Instead of sequential async tasks, we now use a 3-Phase pipeline:
        # Phase 1: Pre-inference matching and gating (Fast/CPU)
        # Phase 2: Batch NLI inference (Dense/GPU-efficient)
        # Phase 3: Final scoring and statistics

        # Phase 1: Preprocess all candidates
        candidates = []
        # Hydrate FC candidates
        for claim, article, similarity_score, chunk_texts in factcheck_results:
            pre_res = self._preprocess_result(
                user_claim_norm=user_claim_core_norm,
                claim_entities=claim_entities,
                similarity_score=similarity_score,
                article=article,
                claim_text=claim.claim_text,
                claim_verdict=claim.verdict,
                source_bias=article.source_bias,
                is_factcheck=True,
                is_negated=is_negated,
                chunk_texts=chunk_texts,
            )
            candidates.append({"pre": pre_res, "article": article, "is_fc": True})

        # Hydrate News candidates
        for article, similarity_score, chunk_texts in news_results:
            if (article.type or "").strip().lower() == "fact-check":
                continue
            pre_res = self._preprocess_result(
                user_claim_norm=user_claim_core_norm,
                claim_entities=claim_entities,
                similarity_score=similarity_score,
                article=article,
                claim_text=None,
                claim_verdict=None,
                source_bias=article.source_bias,
                is_factcheck=False,
                is_negated=is_negated,
                chunk_texts=chunk_texts,
            )
            candidates.append({"pre": pre_res, "article": article, "is_fc": False})

        # Phase 2: Batch NLI for those that passed the relevance gate
        valid_for_nli = [c for c in candidates if c["pre"]["meets_relevance_gate"]]

        if valid_for_nli:
            article_snippets = [c["pre"]["nli_text"] for c in valid_for_nli]
            # Use ORIGINAL user claim for NLI to avoid double-negative logic traps
            claims_to_test = [user_claim]

            # Run batch classification - PREMISE is article, HYPOTHESIS is claim
            loop = asyncio.get_event_loop()
            nli_batch_results = await loop.run_in_executor(
                self.executor,
                self.nli_service.classify_nli_batch,
                article_snippets,
                claims_to_test,
            )

            # Assign NLI results back to candidates
            for i, nli_res in enumerate(nli_batch_results):
                valid_for_nli[i]["nli"] = nli_res
        else:
            nli_batch_results = []

        # Phase 3: Post-processing and scoring
        results: list[ArticleResultModel] = []
        for cand in candidates:
            # If it didn't pass the gate, we still postprocess it to get the skip reasons
            nli_data = cand.get("nli", (NLILabel.NEUTRAL, 0.0, 0.0))
            post_res = self._postprocess_result(
                result=cand["pre"],
                nli_label=nli_data[0],
                nli_score=nli_data[1],
                nli_uncertainty=nli_data[2],
                article=cand["article"],
                is_factcheck=cand["is_fc"],
                is_negated=is_negated,
            )
            results.append(post_res)

        # Filter and sort results
        skipped: list[ArticleResultModel] = []
        filtered_results: list[ArticleResultModel] = []

        for result in results:
            if len(result.skip_reason) > 0:
                skipped.append(result)
            elif result.source_type == "fact_check" and result.found_verdict in (
                "UNKNOWN",
                None,
            ):
                result.skip_reason.append(
                    "Skipped: Fact check has UNKNOWN verdict per strict rules."
                )
                skipped.append(result)
            else:
                filtered_results.append(result)

        # Sort logic:
        # 1. Results with NLI scores first
        # 2. Sort by combined relevance score (highest first)
        # 3. Fact-checks as secondary tie-breaker (only if they are actually relevant)
        filtered_results.sort(
            key=lambda x: (
                0 if x.nli_result else 1,
                -x.combined_relevance_score,
                0 if x.found_claim else 1,
            )
        )

        # Prevent low-relevance results from skewing the final verdict,
        # while still allowing the UI to display everything for manual review.
        # But wait! We NO LONGER restrict the scoring array, because Power-4
        # weighting inherently phases out noise. However, we DO need the 'limit'
        # to mark 'is_aggregated' for the UI display.
        results_for_scoring = filtered_results

        limit = (
            aggregation_limit
            if aggregation_limit is not None
            else self.AGGREGATION_LIMIT
        )

        if limit is not None and limit != "":
            try:
                limit = int(limit)
            except (ValueError, TypeError):
                limit = None

        # Mark which ones were heavily favored for UI highlighting
        if limit is not None and limit > 0:
            for i, res in enumerate(filtered_results):
                res.is_aggregated = i < limit
        else:
            for res in filtered_results:
                res.is_aggregated = True

        stats = self.stats_service.calculate_stats(results_for_scoring)

        overall_verdict = stats["overall_verdict"]

        # NOTE: Overall sign-flip for is_negated is now handled at the individual result level
        # inside process_result_async to ensure UI consistency (Evidence shows SUPPORT for negated intent).
        overall_verdict = stats["overall_verdict"]

        return {
            "skipped": skipped,
            "results": filtered_results,
            "overall_verdict": overall_verdict,
            "bias_divergence": stats["bias_divergence"],
            "truth_confidence_score": stats["truth_confidence_score"],
            "bias_consistency": stats["bias_consistency"],
            "is_negated": is_negated,  # Expose for debugging/frontend use
        }

    def _preprocess_result(
        self,
        user_claim_norm: str,
        claim_entities: list[str],
        similarity_score: float,
        article: Article,
        claim_text: str | None,
        claim_verdict: Verdict | None,
        source_bias: SourceBias,
        is_factcheck: bool,
        is_negated: bool = False,
        chunk_texts: str | None = None,
    ) -> dict:
        """
        Phase 1: Pre-inference matching and gating logic.
        Handles entity overlap, keyword scoring, and relevance thresholds.
        Returns a dict of context for NLI and final processing.
        """
        # For entity matching: use claim text for FC, or article content+title for news
        if is_factcheck and claim_text:
            entity_comparison_text = (claim_text or "") + " " + (chunk_texts or "")
            entity_comparison_title = article.title
        else:
            entity_comparison_text = (article.content or "") + " " + (chunk_texts or "")
            entity_comparison_title = article.title

        entity_match_score = self.calculate_entity_match_score(
            claim_entities, entity_comparison_text, entity_comparison_title
        )
        combined_relevance_score = (similarity_score * self.SEMANTIC_WEIGHT) + (
            entity_match_score * self.ENTITY_WEIGHT
        )

        display_content = (
            article.content[:500] + "..."
            if article.content and len(article.content) > 500
            else (article.content or chunk_texts or "")
        )

        result: dict = {
            "doc_id": article.doc_id,
            "title": article.title,
            "content": display_content,
            "found_claim": claim_text,
            "found_verdict": claim_verdict,
            "publish_date": article.publish_date.isoformat(),
            "source": article.source,
            "url": article.url,
            "similarity_score": round(similarity_score, 4),
            "entity_match_score": round(entity_match_score, 4),
            "combined_relevance_score": round(combined_relevance_score, 4),
            "nli_result": None,
            "verdict": None,
            "skip_reason": [],
            "source_type": "fact_check" if is_factcheck else "news_article",
            "source_bias": source_bias,
            "chunk_texts": chunk_texts,
            "meets_relevance_gate": True,
        }

        requires_specific_match = self.requires_specific_entity_match(claim_entities)
        has_specific_match = self.has_specific_entity_token_match(
            claim_entities, entity_comparison_text, entity_comparison_title
        )

        effective_similarity_threshold = self.RELEVANCE_THRESHOLD
        effective_combined_threshold = self.COMBINED_THRESHOLD
        if has_specific_match and entity_match_score >= 0.3:
            effective_similarity_threshold = -0.5
            effective_combined_threshold = 0.10

        meets_relevance_gate = (
            similarity_score >= effective_similarity_threshold
            and entity_match_score >= self.ENTITY_THRESHOLD
            and combined_relevance_score >= effective_combined_threshold
        )

        # Keyword Gate
        article_text_norm = self.normalize_text(
            entity_comparison_text, strip_punctuation=True
        )
        article_title_norm = self.normalize_text(
            entity_comparison_title, strip_punctuation=True
        )
        content_tokens = self.tokenize_text(article_text_norm)
        title_tokens = self.tokenize_text(article_title_norm)
        article_tokens = content_tokens.union(title_tokens)

        claim_tokens = set(
            t.lower()
            for t in self.tokenize_text(
                self.normalize_text(user_claim_norm, strip_punctuation=True)
            )
        )

        entity_parts = set()
        token_to_entity_map = {}
        for ent in claim_entities:
            ent_tokens = self.tokenize_text(
                self.normalize_text(ent, strip_punctuation=True)
            )
            for t in ent_tokens:
                t_low = t.lower()
                entity_parts.add(t_low)
                if t_low not in token_to_entity_map:
                    token_to_entity_map[t_low] = set()
                token_to_entity_map[t_low].add(ent)

        matched_meaningful = {
            ct
            for ct in {
                t
                for t in claim_tokens
                if t not in COMMON_STOPWORDS and t not in STOP_TITLES
            }
            if any(ct == at or self.is_fuzzy_match(ct, at) for at in article_tokens)
        }

        relevance_points = len(
            [t for t in matched_meaningful if t not in ENTITY_GENERIC_TOKENS]
        )
        entity_token_matches = {
            t
            for t in matched_meaningful
            if t in entity_parts and t not in ENTITY_GENERIC_TOKENS
        }
        topical_matches = {
            t
            for t in matched_meaningful
            if t not in entity_parts and t not in ENTITY_GENERIC_TOKENS
        }
        descriptor_matches = {
            t for t in matched_meaningful if t in ENTITY_GENERIC_TOKENS
        }

        # Ensure meaningful_claim_tokens exists if I used it above
        meaningful_claim_tokens = {
            t
            for t in claim_tokens
            if t not in COMMON_STOPWORDS and t not in STOP_TITLES
        }
        specific_tokens_in_claim = {
            t for t in meaningful_claim_tokens if t not in ENTITY_GENERIC_TOKENS
        }

        matched_full_entities = set()
        for t in entity_token_matches:
            if t in token_to_entity_map:
                matched_full_entities.update(token_to_entity_map[t])
        distinct_entities_matched = len(matched_full_entities)

        specific_tokens_in_claim = (
            {t for t in meaningful_claim_tokens if t not in ENTITY_GENERIC_TOKENS}
            if "meaningful_claim_tokens" in locals()
            else set()
        )

        gate_threshold = (
            min(2, len(specific_tokens_in_claim)) if specific_tokens_in_claim else 1
        )
        has_topical_assertion = (
            len(specific_tokens_in_claim.difference(entity_parts)) > 0
        )
        has_event_marker = any(t in EVENT_MARKERS for t in descriptor_matches)

        is_persona_noise = (
            has_topical_assertion
            and not topical_matches
            and distinct_entities_matched <= 1
            and (relevance_points < 3 or not has_specific_match)
            and not has_event_marker
        )

        if is_persona_noise and distinct_entities_matched == 1:
            matched_ent = (
                list(matched_full_entities)[0] if matched_full_entities else ""
            )
            if len(matched_ent.split()) >= 2:
                is_persona_noise = False

        keyword_match = (relevance_points >= gate_threshold) or (
            has_specific_match and relevance_points >= 1
        )
        if is_persona_noise:
            keyword_match = False

        if (requires_specific_match and not has_specific_match) or not keyword_match:
            meets_relevance_gate = False

        # Additional info for Phase 2
        result.update(
            {
                "meets_relevance_gate": meets_relevance_gate,
                "relevance_points": relevance_points,
                "distinct_entities_matched": distinct_entities_matched,
                "topical_matches": topical_matches,
                "entity_token_matches": entity_token_matches,
                "claim_tokens": claim_tokens,
                "title_tokens": title_tokens,
                "specific_tokens_in_claim": specific_tokens_in_claim,
                "entity_parts": entity_parts,
                "effective_similarity_threshold": effective_similarity_threshold,
                "effective_combined_threshold": effective_combined_threshold,
                "requires_specific_match": requires_specific_match,
                "has_specific_match": has_specific_match,
                "keyword_match": keyword_match,
                "nli_text": (
                    f"{claim_text} {chunk_texts or ''}"
                    if is_factcheck and claim_text
                    else (chunk_texts or article.content)
                ),
            }
        )
        return result

    def _postprocess_result(
        self,
        result: dict,
        nli_label: NLILabel,
        nli_score: float,
        nli_uncertainty: float,
        article: Article,
        is_factcheck: bool,
        is_negated: bool,
    ) -> ArticleResultModel:
        """
        Phase 2: Use batch NLI result to finalize scoring and remarks.
        """
        meets_relevance_gate = result["meets_relevance_gate"]

        # Polarity Guard
        if nli_label != NLILabel.REFUTE:
            if self.is_polarity_mismatch(
                result["claim_tokens"], result["title_tokens"]
            ):
                nli_label, nli_score, nli_uncertainty = NLILabel.REFUTE, 0.8, 0.0

        nli_label_display = nli_label

        result["nli_result"] = {
            "relationship": nli_label_display,
            "relationship_confidence": nli_score,
            "relationship_uncertainty": nli_uncertainty,
            "claim_source": article.source,
            "analyzed_text": self.truncate_at_sentence(result["nli_text"], 200),
        }

        # Context evaluation
        is_strong_context = (
            (is_factcheck and result["similarity_score"] >= 0.55)
            or (result["relevance_points"] >= 3)
            or (
                result["relevance_points"] >= 2
                and (
                    result["distinct_entities_matched"] >= 2
                    or result["topical_matches"]
                )
            )
        )

        # Gates
        if nli_label == NLILabel.NEUTRAL and not is_strong_context:
            result["skip_reason"].append(
                "Article is neutrally related (different event/topic)"
            )
            meets_relevance_gate = False
        elif nli_uncertainty > self.UNCERTAINTY_THRESHOLD and not is_strong_context:
            result["skip_reason"].append(
                f"NLI uncertainty too high (entropy {nli_uncertainty:.2f} > {self.UNCERTAINTY_THRESHOLD})"
            )
            meets_relevance_gate = False

        # Topical Precision
        claim_assertions = result["specific_tokens_in_claim"].difference(
            result["entity_parts"]
        )
        if (
            nli_label == NLILabel.SUPPORT
            and claim_assertions
            and not result["topical_matches"]
        ):
            if not is_strong_context:
                result["skip_reason"].append(
                    f"NLI Support rejected: Evidence lacks topical assertions ({', '.join(list(claim_assertions)[:3])}...)"
                )
                meets_relevance_gate = False
            else:
                nli_score *= 0.5
                result["nli_result"]["relationship_confidence"] = nli_score

        if (
            nli_score < self.NLI_CONFIDENCE_GATE
            and meets_relevance_gate
            and not is_strong_context
        ):
            result["skip_reason"].append(
                f"NLI confidence too low ({nli_score:.2f} < {self.NLI_CONFIDENCE_GATE})"
            )
            meets_relevance_gate = False
        elif meets_relevance_gate:
            verdict_score = self.compute_final_score(
                verdict=(
                    Verdict(result["found_verdict"])
                    if result["found_verdict"]
                    else None
                ),
                source_bias=SourceBias(result["source_bias"]),
                nli_label=nli_label,
                nli_score=nli_score,
                is_factcheck=is_factcheck,
                similarity_score=result["similarity_score"],
                article_content=article.content or "",
                has_topical_match=bool(
                    result["topical_matches"] or result["entity_token_matches"]
                ),
                is_negated=is_negated,
            )
            result["verdict"] = verdict_score

        # Clear verdict if any gate failed
        if not meets_relevance_gate:
            result["verdict"] = None
            if result["requires_specific_match"] and not result["has_specific_match"]:
                result["skip_reason"].append(
                    "Specific name/identifier from claim not found"
                )
            if result["similarity_score"] < result["effective_similarity_threshold"]:
                result["skip_reason"].append(
                    f"Low semantic similarity ({result['similarity_score']})"
                )
            if result["entity_match_score"] < self.ENTITY_THRESHOLD:
                result["skip_reason"].append(
                    f"Key entities not found ({result['entity_match_score']})"
                )
            if (
                result["combined_relevance_score"]
                < result["effective_combined_threshold"]
            ):
                result["skip_reason"].append(
                    f"Balanced relevance score too low ({result['combined_relevance_score']})"
                )
            if not result["keyword_match"]:
                result["skip_reason"].append(
                    f"Insufficient topical overlap (Points: {result['relevance_points']})"
                )
            if not result["skip_reason"]:
                result["skip_reason"].append("Did not meet filtering criteria")

        # Cleanup internal keys before returning model
        final_keys = [f for f in ArticleResultModel.__fields__]
        final_dict = {k: result[k] for k in result if k in final_keys}
        return ArticleResultModel(**final_dict)

    async def process_result_async(
        self,
        user_claim_norm: str,
        claim_entities: list[str],
        similarity_score: float,
        article: Article,
        claim_text: str | None,
        claim_verdict: Verdict | None,
        source_bias: SourceBias,
        is_factcheck: bool,
        is_negated: bool = False,
        chunk_texts: str | None = None,
    ) -> ArticleResultModel:
        """
        Legacy entry point for processing a single result.
        Uses the new phase-based pipeline internally.
        """
        pre_res = self._preprocess_result(
            user_claim_norm=user_claim_norm,
            claim_entities=claim_entities,
            similarity_score=similarity_score,
            article=article,
            claim_text=claim_text,
            claim_verdict=claim_verdict,
            source_bias=source_bias,
            is_factcheck=is_factcheck,
            is_negated=is_negated,
            chunk_texts=chunk_texts,
        )

        nli_label, nli_score, nli_uncertainty = NLILabel.NEUTRAL, 0.0, 0.0
        if pre_res["meets_relevance_gate"]:
            loop = asyncio.get_event_loop()
            nli_label, nli_score, nli_uncertainty = await loop.run_in_executor(
                self.executor,
                self.nli_service.classify_nli,
                pre_res["nli_text"],
                user_claim_norm,
            )

        return self._postprocess_result(
            result=pre_res,
            nli_label=nli_label,
            nli_score=nli_score,
            nli_uncertainty=nli_uncertainty,
            article=article,
            is_factcheck=is_factcheck,
            is_negated=is_negated,
        )

    async def verify_claim_stream_with_stats(self, user_claim: str):
        """
        Streams search hits, then result items (filtered/skipped), then remarks updates.
        """
        user_claim_norm = self.normalize_text(user_claim)
        user_claim_for_matching = self.normalize_text(user_claim, lowercase=False)
        loop = asyncio.get_event_loop()

        # --- Parallel Startup ---
        # 1. Detect negation/stance
        core_claim_text, is_negated = self.detect_claim_stance(user_claim_for_matching)
        user_claim_core_norm = self.normalize_text(core_claim_text)

        # 2. Run embedding and entity extraction in parallel (Gathers expensive ML once)
        claim_embedding, claim_entities = await asyncio.gather(
            loop.run_in_executor(
                self.executor, self.embedding_service.embed_text, core_claim_text
            ),
            loop.run_in_executor(
                self.executor, self.extract_entities, user_claim_for_matching
            ),
        )

        # 3. Stream search hits
        factcheck_task = loop.run_in_executor(
            self.executor,
            self.find_claims_with_articles,
            claim_embedding,
            self.DB_RETRIEVE_LIMIT,
        )
        news_task = loop.run_in_executor(
            self.executor,
            self.find_news_articles,
            claim_embedding,
            self.DB_RETRIEVE_LIMIT,
        )

        factcheck_results, news_results = await asyncio.gather(
            factcheck_task, news_task
        )

        search_hits = []
        for claim, article, _, _ in factcheck_results:
            search_hits.append(
                {
                    "doc_id": article.doc_id,
                    "title": article.title,
                    "url": article.url,
                    "publish_date": article.publish_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": article.source,
                    "source_type": "fact_check",
                }
            )
        for article, _, _ in news_results:
            search_hits.append(
                {
                    "doc_id": article.doc_id,
                    "title": article.title,
                    "url": article.url,
                    "publish_date": article.publish_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": article.source,
                    "source_type": "news_article",
                }
            )
        yield {
            "type": StreamEventType.SEARCH_HITS,
            "hits": search_hits,
        }

        # 2. Stream results (filtered/skipped, remarks pending)
        processed_doc_ids = set()
        tasks = []

        for claim, article, similarity_score, chunk_texts in factcheck_results:
            if article.doc_id not in processed_doc_ids:
                processed_doc_ids.add(article.doc_id)
                tasks.append(
                    self.process_result_async(
                        user_claim_norm=user_claim_core_norm,
                        claim_entities=claim_entities,
                        similarity_score=similarity_score,
                        article=article,
                        claim_text=claim.claim_text,
                        claim_verdict=claim.verdict,
                        source_bias=article.source_bias,
                        is_factcheck=True,
                        is_negated=is_negated,
                        chunk_texts=chunk_texts,
                    )
                )

        for article, similarity_score, chunk_texts in news_results:
            if article.doc_id not in processed_doc_ids:
                processed_doc_ids.add(article.doc_id)
                if (article.type or "").strip().lower() == "fact-check":
                    continue
                tasks.append(
                    self.process_result_async(
                        user_claim_norm=user_claim_core_norm,
                        claim_entities=claim_entities,
                        similarity_score=similarity_score,
                        article=article,
                        claim_text=None,
                        claim_verdict=None,
                        source_bias=article.source_bias,
                        is_factcheck=False,
                        is_negated=is_negated,
                        chunk_texts=chunk_texts,
                    )
                )

        results: list[ArticleResultModel] = []
        remarks_tasks: list[tuple[str, ArticleResultModel]] = []

        for coro in asyncio.as_completed(tasks):
            result: ArticleResultModel = await coro
            # If not skipped, remarks will be generated later
            if not result.skip_reason:
                remarks_tasks.append((result.doc_id, result))
            yield {
                "type": StreamEventType.RESULT,
                "data": result.model_dump(),
                "skipped": bool(result.skip_reason),
            }
            results.append(result)

        # 3. Stream final stats
        # Ensure consistent sorting and aggregation for streaming results
        non_skipped = [r for r in results if not r.skip_reason]
        non_skipped.sort(
            key=lambda x: (
                0 if x.nli_result else 1,
                -x.combined_relevance_score,
                0 if x.found_claim else 1,
            )
        )

        limit = self.AGGREGATION_LIMIT
        aggregated_results = non_skipped[:limit]

        # Mark aggregated for UI identification
        for i, res in enumerate(non_skipped):
            res.is_aggregated = i < limit

        final_stats = self.stats_service.calculate_stats(aggregated_results)
        yield {
            "type": StreamEventType.STATS,
            "total_results": len(results),
            "stats": final_stats,
        }

        # 4. Catch-up: Generate remarks and stream updates
        for doc_id, result in remarks_tasks:
            nli_label = (
                result.nli_result.relationship
                if result.nli_result
                else NLILabel.NEUTRAL
            )
            verdict = result.verdict if result.verdict is not None else 0.0

            remarks = await loop.run_in_executor(
                self.executor,
                self.remarks_generation_service.generate_remarks,
                result.chunk_texts or result.content,
                verdict,
                nli_label,
                not (result.source_type == "fact_check"),
            )

            yield {
                "type": StreamEventType.REMARKS,
                "doc_id": doc_id,
                "remarks": remarks,
            }

        yield {
            "type": StreamEventType.COMPLETE,
        }

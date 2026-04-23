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
from services import (
    EmbeddingService,
    EntityExtractionService,
    NLIService,
    RemarksGenerationService,
    StatsService,
)
import unicodedata
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine
from collections import defaultdict
from databases.verify import VerifyDatabase
import logging
from shared.helpers import get_env_int

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight data containers
# ---------------------------------------------------------------------------


class _TokenSets:
    """Pre-computed token sets shared across gating steps."""

    __slots__ = (
        "claim_tokens",
        "article_tokens",
        "title_tokens",
        "entity_parts",
        "token_to_entity_map",
        "meaningful_claim_tokens",
    )

    def __init__(
        self,
        claim_tokens: set[str],
        article_tokens: set[str],
        title_tokens: set[str],
        entity_parts: set[str],
        token_to_entity_map: dict[str, set[str]],
        meaningful_claim_tokens: set[str],
    ):
        self.claim_tokens = claim_tokens
        self.article_tokens = article_tokens
        self.title_tokens = title_tokens
        self.entity_parts = entity_parts
        self.token_to_entity_map = token_to_entity_map
        self.meaningful_claim_tokens = meaningful_claim_tokens


class _GateResult:
    """Outcome of the relevance-gating pass."""

    __slots__ = (
        "passes",
        "relevance_points",
        "topical_matches",
        "descriptor_matches",
        "entity_token_matches",
        "distinct_entities_matched",
        "has_specific_match",
        "requires_specific_match",
        "skip_reasons",
    )

    def __init__(
        self,
        passes: bool,
        relevance_points: int,
        topical_matches: set[str],
        descriptor_matches: set[str],
        entity_token_matches: set[str],
        distinct_entities_matched: int,
        has_specific_match: bool,
        requires_specific_match: bool,
        skip_reasons: list[str],
    ):
        self.passes = passes
        self.relevance_points = relevance_points
        self.topical_matches = topical_matches
        self.descriptor_matches = descriptor_matches
        self.entity_token_matches = entity_token_matches
        self.distinct_entities_matched = distinct_entities_matched
        self.has_specific_match = has_specific_match
        self.requires_specific_match = requires_specific_match
        self.skip_reasons = skip_reasons


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class VerifyController:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.entity_extraction_service = EntityExtractionService()
        self.nli_service = NLIService()
        self.remarks_generation_service = RemarksGenerationService()
        self.stats_service = StatsService()
        self.db = VerifyDatabase()

        # Score thresholds
        self.RELEVANCE_THRESHOLD = 0.3
        self.ENTITY_THRESHOLD = 0.3
        self.COMBINED_THRESHOLD = 0.4

        # Combined score weights (semantic 70%, entity 30%)
        self.SEMANTIC_WEIGHT = 0.7
        self.ENTITY_WEIGHT = 0.3

        # ----------------------------------------------------------------
        # Threading / concurrency knobs
        #
        # NLI_MAX_THREADS=2 + NLI_TORCH_NUM_THREADS=1 (set in NLIService):
        #   2 NLI jobs run truly in parallel, each pinned to 1 vCPU.
        #   Old config (NLI_MAX_THREADS=1, torch_threads=2) used the same
        #   2 CPUs but serialised all requests at the NLI step.
        #
        # MAX_THREADS=3: shared pool for embedding, entity extraction, and
        #   remarks generation — all lighter than NLI, yield frequently.
        # ----------------------------------------------------------------
        self.MAX_THREADS = get_env_int("VERIFY_MAX_THREADS", 3, min_value=1)
        self.NLI_MAX_THREADS = get_env_int("VERIFY_NLI_MAX_THREADS", 2, min_value=1)

        self.MAX_CONCURRENT_PROCESSES = get_env_int(
            "VERIFY_MAX_CONCURRENT_PROCESSES", 7, min_value=1
        )
        self.PER_REQUEST_CONCURRENCY_LIMIT = get_env_int(
            "VERIFY_PER_REQUEST_CONCURRENCY_LIMIT", 4, min_value=1
        )
        self.AGGREGATION_LIMIT = get_env_int("VERIFY_AGGREGATION_LIMIT", 3, min_value=1)
        self.DB_RETRIEVE_LIMIT = get_env_int(
            "VERIFY_DB_RETRIEVE_LIMIT", 20, min_value=1
        )

        # Separate executors: NLI is CPU-heavy and must not share a pool with
        # lighter tasks that would otherwise queue behind a running NLI job.
        self.executor = ThreadPoolExecutor(
            max_workers=self.MAX_THREADS, thread_name_prefix="verify"
        )
        self.nli_executor = ThreadPoolExecutor(
            max_workers=self.NLI_MAX_THREADS, thread_name_prefix="nli"
        )

        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_PROCESSES)

        self.NLI_CONFIDENCE_GATE = 0.60
        self.UNCERTAINTY_THRESHOLD = 0.80
        self.NEWS_SIMILARITY_THRESHOLD = 0.45

    # -----------------------------------------------------------------------
    # Text utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def normalize_text(
        text: str, lowercase: bool = True, strip_punctuation: bool = False
    ) -> str:
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFKC", text)
        if strip_punctuation:
            text = re.sub(r"[^\w\s-]", " ", text)
        if lowercase:
            text = text.lower()
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def tokenize_text(text: str) -> set[str]:
        return set(re.findall(r"[^\W\d_]{2,}", text))

    @staticmethod
    def truncate_at_sentence(text: str, max_chars: int = 200) -> str:
        cleaned = re.sub(r"\s+", " ", text.strip())
        if len(cleaned) <= max_chars:
            return cleaned
        truncated = cleaned[:max_chars]
        sentence_matches = list(re.finditer(r"[.!?](?=\s+[A-Z0-9]|$)", truncated))
        if sentence_matches:
            return cleaned[: sentence_matches[-1].end()].strip()
        return truncated.rsplit(" ", 1)[0].rstrip() + "..."

    @staticmethod
    def _truncate_content(article: Article, chunk_texts: str | None) -> str:
        """Produce a ≤500-char display string. Extracted to avoid repeating this logic."""
        content = article.content
        if content and len(content) > 500:
            return content[:500] + "..."
        return content or chunk_texts or ""

    @staticmethod
    def build_chunk_text(
        chunks: list[ArticleChunk], max_chars: int = 800, max_chunks: int = 3
    ) -> str | None:
        if not chunks:
            return None
        parts: list[str] = []
        used_chars = 0
        for chunk in chunks:
            if len(parts) >= max_chunks or used_chars >= max_chars:
                break
            content = chunk.chunk_content.strip()
            if not content:
                continue
            remaining = max_chars - used_chars
            snippet = content[:remaining].rsplit(" ", 1)[0].strip()
            if not snippet:
                continue
            parts.append(snippet)
            used_chars += len(snippet) + 1
        return "\n\n".join(parts).strip() or None

    # -----------------------------------------------------------------------
    # Fuzzy / entity matching utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def is_fuzzy_match(t1: str, t2: str) -> bool:
        if t1 == t2:
            return True
        t1_l, t2_l = t1.lower(), t2.lower()
        for group in DEMONYM_GROUPS:
            if t1_l in group and t2_l in group:
                return True
        if len(t1) < 3 or len(t2) < 3:
            return False
        for suffix in COMMON_PLURAL_SUFFIXES:
            if t1 + suffix == t2 or t2 + suffix == t1:
                return True
            if suffix == "ies":
                if (t1.endswith("y") and t1[:-1] + "ies" == t2) or (
                    t2.endswith("y") and t2[:-1] + "ies" == t1
                ):
                    return True
        s, l = (t1_l, t2_l) if len(t1_l) < len(t2_l) else (t2_l, t1_l)
        if len(s) >= 4 and l.startswith(s):
            return True
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
        if not claim_entities:
            return 0.0

        text_norm = self.normalize_text(text, strip_punctuation=True)
        title_norm = self.normalize_text(article_title, strip_punctuation=True)
        comparison_tokens = self.tokenize_text(text_norm) | self.tokenize_text(
            title_norm
        )

        matches = 0.0
        total_weight = 0.0

        for entity in claim_entities:
            entity_lower = self.normalize_text(entity, strip_punctuation=True)
            entity_tokens_all = self.tokenize_text(entity_lower)
            specific_entity_tokens = [
                t
                for t in entity_tokens_all
                if t not in ENTITY_GENERIC_TOKENS and len(t) >= 3
            ]
            entity_weight = (
                0.25 if entity_tokens_all and not specific_entity_tokens else 1.0
            )
            total_weight += entity_weight

            # Exact whole-word match
            if re.search(
                r"\b" + re.escape(entity_lower) + r"\b", text_norm
            ) or re.search(r"\b" + re.escape(entity_lower) + r"\b", title_norm):
                matches += entity_weight
                continue

            # Partial token match fallback
            if specific_entity_tokens:
                found_partial = any(
                    token in comparison_tokens
                    or any(self.is_fuzzy_match(token, ct) for ct in comparison_tokens)
                    for token in specific_entity_tokens
                )
                if found_partial:
                    matches += 0.7 * entity_weight

        return matches / total_weight if total_weight > 0 else 0.0

    def requires_specific_entity_match(self, claim_entities: list[str]) -> bool:
        has_generic = has_specific = False
        for entity in claim_entities:
            for token in self.tokenize_text(self.normalize_text(entity)):
                if token in ENTITY_GENERIC_TOKENS:
                    has_generic = True
                elif len(token) >= 3:
                    has_specific = True
        return has_generic and has_specific

    def has_specific_entity_token_match(
        self, claim_entities: list[str], text: str, article_title: str = ""
    ) -> bool:
        comparison_tokens = self.tokenize_text(
            self.normalize_text(text, strip_punctuation=True)
        ) | self.tokenize_text(
            self.normalize_text(article_title, strip_punctuation=True)
        )
        for entity in claim_entities:
            specific_tokens = [
                t
                for t in self.tokenize_text(
                    self.normalize_text(entity, strip_punctuation=True)
                )
                if t not in ENTITY_GENERIC_TOKENS and len(t) >= 3
            ]
            if any(
                any(self.is_fuzzy_match(st, ct) for ct in comparison_tokens)
                for st in specific_tokens
            ):
                return True
        return False

    @staticmethod
    def is_polarity_mismatch(tokens1: set[str], tokens2: set[str]) -> bool:
        t1_low = {t.lower() for t in tokens1}
        t2_low = {t.lower() for t in tokens2}
        for a, b in ANTONYM_PAIRS:
            if (a in t1_low and b in t2_low) or (b in t1_low and a in t2_low):
                return True
        has_neg1 = any(n in t1_low for n in NEGATION_TOKENS)
        has_neg2 = any(n in t2_low for n in NEGATION_TOKENS)
        return has_neg1 != has_neg2

    # -----------------------------------------------------------------------
    # Stance / negation detection
    # -----------------------------------------------------------------------

    def detect_claim_stance(self, claim: str) -> tuple[str, bool]:
        """
        Strips leading/inline negation markers and returns (core_claim, is_negated).
        """
        claim_lower = claim.lower().strip()

        for phrase in NEGATION_PHRASES:
            if claim_lower.startswith(phrase):
                core = claim[len(phrase) :].strip()
                return (core[0].upper() + core[1:] if core else core), True

        for pattern, _ in NEGATION_WORD_PATTERNS:
            match = re.search(pattern, claim, flags=re.IGNORECASE)
            if match:
                core = re.sub(
                    pattern, r"\1" if match.groups() else "", claim, flags=re.IGNORECASE
                )
                core = re.sub(r"\s+", " ", core).strip()
                return (core[0].upper() + core[1:] if core else core), True

        return claim, False

    # -----------------------------------------------------------------------
    # Entity extraction
    # -----------------------------------------------------------------------

    def extract_entities(self, text: str) -> list[str]:
        entities_with_label = self.entity_extraction_service.extract_entities(text)
        seen: set[str] = set()
        result: list[str] = []
        for entity, _ in entities_with_label:
            if not entity:
                continue
            norm = self.normalize_text(entity)
            if norm and norm not in seen:
                seen.add(norm)
                result.append(entity)
        return result

    # -----------------------------------------------------------------------
    # DB / retrieval helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_doc_ids(rows: list[tuple]) -> set[str]:
        """Extract unique doc_ids from a list of (ORM object, distance) tuples."""
        return {row[0].doc_id for row in rows}

    def _get_chunk_map(
        self, embedding: list[float], doc_ids: set[str], top_n: int = 3
    ) -> dict[str, list[tuple[ArticleChunk, float]]]:
        """Returns {doc_id: [(chunk, similarity), ...]} sorted best-first, capped at top_n."""
        chunk_map: dict[str, list[tuple[ArticleChunk, float]]] = defaultdict(list)
        for chunk, distance in self.db.find_similar_chunks_from_doc_ids(
            embedding, doc_ids
        ):
            chunk_map[chunk.doc_id].append((chunk, 1 - distance))

        # NOTE: Sort ascending by similarity (lowest first) to match original calibrated behavior.
        # The system's gates and NLI context were tuned against this ordering — do not change
        # without re-calibrating thresholds and re-running accuracy tests.
        for doc_id in chunk_map:
            chunk_map[doc_id] = sorted(chunk_map[doc_id], key=lambda x: x[1])[:top_n]

        return chunk_map

    def _get_article_map(self, doc_ids: set[str]) -> dict[str, Article]:
        return {a.doc_id: a for a in self.db.find_articles_from_doc_ids(doc_ids)}

    def find_claims_with_articles(
        self,
        embedding: list[float],
        top_k: int = 20,
        exclude_doc_ids: list[str] | None = None,
    ) -> list[tuple[Claim, Article, float, str | None]]:
        """
        Returns top_k fact-check claims paired with their articles and best chunk text.
        Excludes doc_ids in exclude_doc_ids.
        """
        similar_claims = self.db.find_similar_claims(embedding, top_k)

        # Fetch chunks/articles for ALL retrieved doc_ids first (matches original DB call order),
        # then filter excluded ones during iteration.
        doc_ids = self._extract_doc_ids(similar_claims)
        article_map = self._get_article_map(doc_ids)
        chunk_map = self._get_chunk_map(embedding, doc_ids)

        exclude = set(exclude_doc_ids or [])
        results: list[tuple[Claim, Article, float, str | None]] = []
        for claim, distance in similar_claims:
            if claim.doc_id in exclude:
                continue
            article = article_map.get(claim.doc_id)
            if article is None:
                continue
            chunks = chunk_map.get(claim.doc_id, [])
            chunk_text = (
                self.build_chunk_text([c for c, _ in chunks]) if chunks else None
            )
            results.append((claim, article, 1 - distance, chunk_text))

        return results

    def find_news_articles(
        self, embedding: list[float], top_k: int = 20
    ) -> list[tuple[Article, float, str | None]]:
        """
        Returns top_k news articles ranked by their best chunk similarity.
        """
        chunk_results = self.db.find_similar_chunks(embedding, top_k)
        doc_ids = self._extract_doc_ids(chunk_results)
        articles = self.db.find_articles_from_doc_ids(doc_ids)
        chunk_map = self._get_chunk_map(embedding, doc_ids)

        results: list[tuple[Article, float, str | None]] = []
        for article in articles:
            chunks = chunk_map.get(article.doc_id, [])
            if not chunks:
                results.append((article, 0.0, None))
                continue
            top_similarity = chunks[0][1]
            chunk_text = self.build_chunk_text([c for c, _ in chunks])
            results.append((article, top_similarity, chunk_text))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # -----------------------------------------------------------------------
    # Token set pre-computation
    # -----------------------------------------------------------------------

    def _build_token_sets(
        self,
        user_claim_norm: str,
        claim_entities: list[str],
        article_comparison_text: str,
        article_title: str,
    ) -> _TokenSets:
        """Pre-compute all token sets used by gating and scoring."""
        article_text_norm = self.normalize_text(
            article_comparison_text, strip_punctuation=True
        )
        article_title_norm = self.normalize_text(article_title, strip_punctuation=True)
        claim_norm_stripped = self.normalize_text(
            user_claim_norm, strip_punctuation=True
        )

        content_tokens = self.tokenize_text(article_text_norm)
        title_tokens = self.tokenize_text(article_title_norm)
        claim_tokens = {t.lower() for t in self.tokenize_text(claim_norm_stripped)}

        entity_parts: set[str] = set()
        token_to_entity_map: dict[str, set[str]] = {}
        for ent in claim_entities:
            ent_norm = self.normalize_text(ent, strip_punctuation=True)
            for t in self.tokenize_text(ent_norm):
                t_low = t.lower()
                entity_parts.add(t_low)
                token_to_entity_map.setdefault(t_low, set()).add(ent)

        meaningful_claim_tokens = {
            t
            for t in claim_tokens
            if t not in COMMON_STOPWORDS and t not in STOP_TITLES
        }

        return _TokenSets(
            claim_tokens=claim_tokens,
            article_tokens=content_tokens | title_tokens,
            title_tokens=title_tokens,
            entity_parts=entity_parts,
            token_to_entity_map=token_to_entity_map,
            meaningful_claim_tokens=meaningful_claim_tokens,
        )

    # -----------------------------------------------------------------------
    # Gating
    # -----------------------------------------------------------------------

    def _score_gate(
        self,
        similarity_score: float,
        entity_match_score: float,
        combined_relevance_score: float,
        has_specific_match: bool,
        requires_specific_match: bool,
    ) -> tuple[bool, float, float]:
        """
        Returns (passes, effective_similarity_threshold, effective_combined_threshold).
        Relaxes thresholds when we have a confirmed specific entity match.
        The is_factcheck-specific entity threshold (0.45) was removed after analysis showed
        99% of FC evidence already scores >= 0.5, making the extra gate a no-op.
        """
        effective_sim = self.RELEVANCE_THRESHOLD
        effective_combined = self.COMBINED_THRESHOLD

        if has_specific_match and entity_match_score >= self.ENTITY_THRESHOLD:
            effective_sim = -0.5  # effectively disabled
            effective_combined = 0.10

        passes = (
            (not requires_specific_match or has_specific_match)
            and similarity_score >= effective_sim
            and entity_match_score >= self.ENTITY_THRESHOLD
            and combined_relevance_score >= effective_combined
        )
        return passes, effective_sim, effective_combined

    def _keyword_gate(
        self, ts: _TokenSets, has_specific_match: bool = False
    ) -> _GateResult:
        """
        Point-based keyword gate: ensures topical overlap between claim and evidence.
        Returns a _GateResult with all intermediate statistics.
        """
        article_tokens = ts.article_tokens

        # Fuzzy-aware meaningful match
        matched_meaningful: set[str] = set()
        for ct in ts.meaningful_claim_tokens:
            if ct in article_tokens:
                matched_meaningful.add(ct)
            else:
                for at in article_tokens:
                    if self.is_fuzzy_match(ct, at):
                        matched_meaningful.add(ct)
                        break

        relevance_points = len(
            [t for t in matched_meaningful if t not in ENTITY_GENERIC_TOKENS]
        )

        topical_matches = {
            t
            for t in matched_meaningful
            if t not in ts.entity_parts and t not in ENTITY_GENERIC_TOKENS
        }
        descriptor_matches = {
            t for t in matched_meaningful if t in ENTITY_GENERIC_TOKENS
        }
        entity_token_matches = {
            t
            for t in matched_meaningful
            if t in ts.entity_parts and t not in ENTITY_GENERIC_TOKENS
        }

        matched_full_entities: set[str] = set()
        for t in entity_token_matches:
            matched_full_entities.update(ts.token_to_entity_map.get(t, set()))
        distinct_entities_matched = len(matched_full_entities)

        # Gate threshold: need at least 2 meaningful matches for specific claims
        specific_tokens_in_claim = {
            t for t in ts.meaningful_claim_tokens if t not in ENTITY_GENERIC_TOKENS
        }
        gate_threshold = (
            min(2, len(specific_tokens_in_claim)) if specific_tokens_in_claim else 1
        )

        # Persona precision guard — mirrors original logic exactly, including has_specific_match
        has_topical_assertion = (
            len(specific_tokens_in_claim.difference(ts.entity_parts)) > 0
        )
        has_event_marker = any(t in EVENT_MARKERS for t in descriptor_matches)
        is_persona_noise = (
            has_topical_assertion
            and not topical_matches
            and distinct_entities_matched <= 1
            and (relevance_points < 3 or not has_specific_match)
            and not has_event_marker
        )

        # Exception: multi-word matched entity is unlikely to be noise
        if is_persona_noise and distinct_entities_matched == 1:
            matched_ent = next(iter(matched_full_entities), "")
            if len(matched_ent.split()) >= 2:
                is_persona_noise = False

        # Coverage bypass: confirmed specific entity match allows 1-point pass (unless persona noise)
        keyword_match = (
            (relevance_points >= gate_threshold)
            or (has_specific_match and relevance_points >= 1)
        ) and not is_persona_noise

        return _GateResult(
            passes=keyword_match,
            relevance_points=relevance_points,
            topical_matches=topical_matches,
            descriptor_matches=descriptor_matches,
            entity_token_matches=entity_token_matches,
            distinct_entities_matched=distinct_entities_matched,
            has_specific_match=has_specific_match,
            requires_specific_match=False,
            skip_reasons=[],
        )

    # -----------------------------------------------------------------------
    # NLI
    # -----------------------------------------------------------------------

    def _build_nli_context(
        self,
        is_factcheck: bool,
        claim_text: str | None,
        chunk_texts: str | None,
        article_content: str | None,
    ) -> str:
        """
        Assemble the NLI premise string.
        For fact-checks we use the matched claim text as the primary premise and
        append chunk_texts as supporting context (reverted from strict claim-only,
        which had zero accuracy effect but reduced context quality for ambiguous claims).
        For news we fall back to chunks then full article content.
        """
        if is_factcheck and claim_text:
            nli_text = claim_text
            if chunk_texts:
                nli_text += f" {chunk_texts}"
            return nli_text
        return chunk_texts or article_content or ""

    def _is_strong_context(
        self,
        is_factcheck: bool,
        similarity_score: float,
        relevance_points: int,
        distinct_entities_matched: int,
        topical_matches: set[str],
    ) -> bool:
        return (
            (is_factcheck and similarity_score >= 0.55)
            or relevance_points >= 3
            or (
                relevance_points >= 2
                and (distinct_entities_matched >= 2 or bool(topical_matches))
            )
        )

    async def _run_nli(
        self,
        nli_text: str,
        user_claim_norm: str,
        claim_tokens: set[str],
        title_tokens: set[str],
    ) -> tuple[NLILabel, float, float]:
        """Run NLI via the dedicated executor and apply polarity guard."""
        loop = asyncio.get_running_loop()
        nli_label, nli_score, nli_uncertainty = await loop.run_in_executor(
            self.nli_executor,
            self.nli_service.classify_nli,
            nli_text,
            user_claim_norm,
        )

        # Polarity guard: override NLI if tokens directly contradict
        if nli_label != NLILabel.REFUTE and self.is_polarity_mismatch(
            claim_tokens, title_tokens
        ):
            nli_label = NLILabel.REFUTE
            nli_score = 0.8
            nli_uncertainty = 0.0

        return nli_label, nli_score, nli_uncertainty

    def _apply_nli_gates(
        self,
        nli_label: NLILabel,
        nli_score: float,
        nli_uncertainty: float,
        is_strong_context: bool,
        claim_assertions: set[str],
        topical_matches: set[str],
    ) -> tuple[bool, float, list[str]]:
        """
        Apply uncertainty, topical-precision, and confidence gates after NLI.
        Returns (still_passes, adjusted_nli_score, extra_skip_reasons).
        """
        skip_reasons: list[str] = []
        passes = True

        if nli_label == NLILabel.NEUTRAL and not is_strong_context:
            skip_reasons.append("Article is neutrally related (different event/topic)")
            passes = False

        elif nli_uncertainty > self.UNCERTAINTY_THRESHOLD and not is_strong_context:
            skip_reasons.append(
                f"NLI uncertainty too high (entropy {nli_uncertainty:.2f} > {self.UNCERTAINTY_THRESHOLD})"
            )
            passes = False

        if (
            passes
            and nli_label == NLILabel.SUPPORT
            and claim_assertions
            and not topical_matches
        ):
            if not is_strong_context:
                skip_reasons.append(
                    f"NLI Support rejected: Evidence lacks topical assertions "
                    f"({', '.join(list(claim_assertions)[:3])}...) mentioned in claim"
                )
                passes = False
            else:
                nli_score = round(nli_score * 0.5, 4)

        if passes and nli_score < self.NLI_CONFIDENCE_GATE and not is_strong_context:
            skip_reasons.append(
                f"NLI confidence too low ({nli_score:.2f} < {self.NLI_CONFIDENCE_GATE}) — unreliable signal"
            )
            passes = False

        return passes, nli_score, skip_reasons

    # -----------------------------------------------------------------------
    # Final score computation
    # -----------------------------------------------------------------------

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
    ) -> float:
        """
        Returns a signed score in [-1, 1]:
            +1 = strong support for user claim (true)
            -1 = strong refutation (false)
            0 = neutral / inconclusive
        """

        if not is_factcheck:
            # Base score reduced from 0.75 → 0.50 to prevent news articles from
            # overriding strong FC signals. News evidence has no verdict anchor,
            # so its influence on the final aggregation should be secondary to FC.
            base = {NLILabel.SUPPORT: 0.50, NLILabel.REFUTE: -0.50}.get(nli_label, 0.0)
            bias_weight = (
                SOURCE_BIAS_WEIGHT_MAP.get(source_bias, 0.7) if source_bias else 0.7
            )
            return round(base * (0.5 + nli_score * 0.5) * bias_weight, 2)

        if verdict is None or source_bias is None:
            return 0.0

        verdict_weight = VERDICT_WEIGHT_MAP.get(verdict, 0.5)
        bias_weight = SOURCE_BIAS_WEIGHT_MAP.get(source_bias, 0.7)
        confidence_factor = nli_score if nli_score >= 0.55 else nli_score * 0.5

        if nli_label == NLILabel.REFUTE:
            nli_label_weight = NLI_LABEL_WEIGHT_MAP.get(nli_label, -1.0)
            if verdict_weight < 0:
                # Double-negative guard: FALSE article + REFUTE
                nli_label_weight = 1.0 if has_topical_match else -1.0

            return round(
                confidence_factor * bias_weight * verdict_weight * nli_label_weight, 2
            )

        nli_label_weight = NLI_LABEL_WEIGHT_MAP.get(nli_label, 0.5)
        raw_score = confidence_factor * bias_weight * verdict_weight * nli_label_weight

        # Dampen AI/video/social-media debunks with low NLI confidence
        if is_factcheck and nli_score < 0.95:
            content_lower = article_content.lower()
            if any(k in content_lower for k in DAMP_KEYWORDS):
                raw_score *= 0.7

        return round(raw_score, 2)

    def calculate_stats(self, evidences: list[ArticleResultModel]):
        """Centralize usage of stats_service's calculate_stats via this function."""
        return self.stats_service.calculate_stats(evidences)

    # -----------------------------------------------------------------------
    # Core per-result processor
    # -----------------------------------------------------------------------

    async def _gather_with_limit(self, tasks, limit=4):
        sem = asyncio.Semaphore(limit)

        async def _wrap(coro):
            async with sem:
                return await coro

        return list(await asyncio.gather(*[_wrap(t) for t in tasks]))

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
        Process a single evidence item through the full scoring pipeline:
            1. Entity match + combined relevance score
            2. Score gate (similarity / entity / combined thresholds)
            3. Keyword gate (point-based topical overlap)
            4. NLI classification
            5. NLI uncertainty / topical-precision / confidence gates
            6. Final score computation + negation flip
        """
        # Compute display content once upfront — avoids the `"display_content" in dir()`
        # anti-pattern from the original (checking local scope via dir() is undefined for locals).
        display_content = self._truncate_content(article, chunk_texts)

        # --- 1. Build comparison strings ---
        if is_factcheck and claim_text:
            comparison_text = (claim_text or "") + " " + (chunk_texts or "")
        else:
            comparison_text = (article.content or "") + " " + (chunk_texts or "")
        comparison_title = article.title or ""

        # Shared base fields reused across all return paths
        base_fields = dict(
            doc_id=article.doc_id,
            title=article.title,
            content=display_content,
            found_claim=claim_text,
            found_verdict=claim_verdict,
            publish_date=article.publish_date.isoformat(),
            source=article.source,
            url=article.url,
            source_bias=source_bias,
            chunk_texts=chunk_texts,
        )

        # --- 2. Early exit for low-similarity news articles ---
        if not is_factcheck and similarity_score < self.NEWS_SIMILARITY_THRESHOLD:
            return ArticleResultModel(
                **base_fields,
                similarity_score=round(similarity_score, 4),
                entity_match_score=0.0,
                combined_relevance_score=0.0,
                nli_result=None,
                verdict=None,
                skip_reason=[
                    f"News similarity too low ({similarity_score:.3f} < {self.NEWS_SIMILARITY_THRESHOLD})"
                ],
                source_type="news_article",
            )

        # --- 3. Scores ---
        entity_match_score = self.calculate_entity_match_score(
            claim_entities, comparison_text, comparison_title
        )
        combined_relevance_score = round(
            similarity_score * self.SEMANTIC_WEIGHT
            + entity_match_score * self.ENTITY_WEIGHT,
            4,
        )

        # --- 4. Build shared token sets (computed once, reused everywhere) ---
        ts = self._build_token_sets(
            user_claim_norm, claim_entities, comparison_text, comparison_title
        )

        # --- 5. Gate checks ---
        requires_specific = self.requires_specific_entity_match(claim_entities)
        has_specific = self.has_specific_entity_token_match(
            claim_entities, comparison_text, comparison_title
        )
        score_passes, eff_sim, eff_combined = self._score_gate(
            similarity_score,
            entity_match_score,
            combined_relevance_score,
            has_specific,
            requires_specific,
        )
        keyword_gate = self._keyword_gate(ts, has_specific_match=has_specific)

        skip_reasons: list[str] = []
        meets_gate = score_passes and keyword_gate.passes
        nli_result_dict = None
        verdict_score = None

        # --- 6. NLI (only if gates pass) ---
        if meets_gate:
            nli_text = self._build_nli_context(
                is_factcheck, claim_text, chunk_texts, article.content
            )
            nli_label, nli_score, nli_uncertainty = await self._run_nli(
                nli_text, user_claim_norm, ts.claim_tokens, ts.title_tokens
            )

            # Display label flip for negated claims (UX only — scoring always uses raw label)
            nli_label_display = nli_label
            if is_negated:
                if nli_label == NLILabel.SUPPORT:
                    nli_label_display = NLILabel.REFUTE
                elif nli_label == NLILabel.REFUTE:
                    nli_label_display = NLILabel.SUPPORT

            nli_result_dict = {
                "relationship": nli_label_display,
                "relationship_confidence": nli_score,
                "relationship_uncertainty": nli_uncertainty,
                "claim_source": article.source,
                "analyzed_text": self.truncate_at_sentence(nli_text, 200),
            }

            # --- NLI gates ---
            specific_tokens_in_claim = {
                t for t in ts.meaningful_claim_tokens if t not in ENTITY_GENERIC_TOKENS
            }
            claim_assertions = specific_tokens_in_claim.difference(ts.entity_parts)
            strong_ctx = self._is_strong_context(
                is_factcheck,
                similarity_score,
                keyword_gate.relevance_points,
                keyword_gate.distinct_entities_matched,
                keyword_gate.topical_matches,
            )
            nli_passes, nli_score, nli_skip = self._apply_nli_gates(
                nli_label,
                nli_score,
                nli_uncertainty,
                strong_ctx,
                claim_assertions,
                keyword_gate.topical_matches,
            )
            skip_reasons.extend(nli_skip)
            meets_gate = nli_passes

            # Update possibly-dampened confidence in the result dict
            nli_result_dict["relationship_confidence"] = nli_score

            # --- Final score ---
            if meets_gate:
                verdict_score = self.compute_final_score(
                    verdict=Verdict(claim_verdict) if claim_verdict else None,
                    source_bias=source_bias,
                    nli_label=nli_label,
                    nli_score=nli_score,
                    is_factcheck=is_factcheck,
                    similarity_score=similarity_score,
                    article_content=article.content or "",
                    has_topical_match=bool(keyword_gate.topical_matches),
                )
                if is_negated and verdict_score != 0:
                    verdict_score = -verdict_score

        # --- 7. Populate skip reasons if gate failed ---
        if not meets_gate and not skip_reasons:
            if requires_specific and not has_specific:
                skip_reasons.append(
                    "Specific name/identifier from claim not found in matched result"
                )
            if similarity_score < eff_sim:
                skip_reasons.append(
                    f"Low semantic similarity ({similarity_score:.3f} < {eff_sim})"
                )
            if entity_match_score < self.ENTITY_THRESHOLD:
                skip_reasons.append(
                    f"Key entities not found ({entity_match_score:.3f} < {self.ENTITY_THRESHOLD})"
                )
            if combined_relevance_score < eff_combined:
                skip_reasons.append(
                    f"Balanced relevance score too low ({combined_relevance_score:.3f} < {eff_combined})"
                )
            if not keyword_gate.passes:
                skip_reasons.append(
                    f"Insufficient topical overlap (Points: {keyword_gate.relevance_points}, "
                    f"Topical: {len(keyword_gate.topical_matches)}, "
                    f"Descriptor: {len(keyword_gate.descriptor_matches)}, "
                    f"Entity Tokens: {keyword_gate.distinct_entities_matched})"
                )
            if not skip_reasons:
                skip_reasons.append("Did not meet filtering criteria")

        return ArticleResultModel(
            **base_fields,
            similarity_score=round(similarity_score, 4),
            entity_match_score=round(entity_match_score, 4),
            combined_relevance_score=combined_relevance_score,
            nli_result=nli_result_dict,
            verdict=verdict_score,
            skip_reason=skip_reasons,
            source_type="fact_check" if is_factcheck else "news_article",
        )

    # -----------------------------------------------------------------------
    # Shared pipeline core (used by both REST and WS entry points)
    # -----------------------------------------------------------------------

    async def _prepare_pipeline_tasks(
        self,
        user_claim: str,
        exclude_doc_ids: list[str] | None = None,
        exclude_articles: bool = False,
    ) -> tuple[
        str,
        str,
        bool,
        list[Coroutine[Any, Any, ArticleResultModel]],
        list[dict[str, Any]],
    ]:
        """
        Build per-evidence processing tasks once so callers can either await all
        results (REST) or consume them as they complete (WebSocket).

        Returns:
            (user_claim_norm, user_claim_core_norm, is_negated, tasks, search_hits)
        """
        user_claim_for_matching = self.normalize_text(user_claim, lowercase=False)
        core_claim_text, is_negated = self.detect_claim_stance(user_claim_for_matching)
        user_claim_norm = self.normalize_text(user_claim_for_matching)
        user_claim_core_norm = self.normalize_text(core_claim_text)

        loop = asyncio.get_running_loop()
        claim_embedding, claim_entities = await asyncio.gather(
            loop.run_in_executor(
                self.executor, self.embedding_service.embed_text, core_claim_text
            ),
            loop.run_in_executor(
                self.executor, self.extract_entities, user_claim_for_matching
            ),
        )

        processed_doc_ids: set[str] = set()
        tasks: list[Coroutine[Any, Any, ArticleResultModel]] = []
        search_hits: list[dict[str, Any]] = []

        def _hit(article: Article, source_type: str) -> dict:
            return {
                "doc_id": article.doc_id,
                "title": article.title,
                "url": article.url,
                "publish_date": article.publish_date.strftime("%Y-%m-%d %H:%M:%S"),
                "source": article.source,
                "source_type": source_type,
            }

        for (
            claim,
            article,
            similarity_score,
            chunk_texts,
        ) in self.find_claims_with_articles(
            claim_embedding, self.DB_RETRIEVE_LIMIT, exclude_doc_ids=exclude_doc_ids
        ):
            search_hits.append(_hit(article, "fact_check"))
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

        if not exclude_articles:
            for article, similarity_score, chunk_texts in self.find_news_articles(
                claim_embedding, self.DB_RETRIEVE_LIMIT
            ):
                search_hits.append(_hit(article, "news_article"))
                if article.doc_id in processed_doc_ids:
                    continue
                if (article.type or "").strip().lower() == "fact-check":
                    continue
                processed_doc_ids.add(article.doc_id)
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

        # deduplicate final hits
        unique_hits = []
        seen_hit_ids = set()
        for hit in search_hits:
            if hit["doc_id"] not in seen_hit_ids:
                seen_hit_ids.add(hit["doc_id"])
                unique_hits.append(hit)

        return (
            user_claim_norm,
            user_claim_core_norm,
            is_negated,
            tasks,
            unique_hits,
        )

    async def _run_pipeline(
        self,
        user_claim: str,
        exclude_doc_ids: list[str] | None = None,
        exclude_articles: bool = False,
    ) -> tuple[str, str, bool, list[ArticleResultModel]]:
        """
        Embed → extract entities → retrieve evidences → process all results.

        Returns:
            (user_claim_norm, user_claim_core_norm, is_negated, all_results)
            where all_results mixes passed and skipped ArticleResultModels.
        """
        user_claim_norm, user_claim_core_norm, is_negated, tasks, _ = (
            await self._prepare_pipeline_tasks(
                user_claim,
                exclude_doc_ids=exclude_doc_ids,
                exclude_articles=exclude_articles,
            )
        )
        all_results: list[ArticleResultModel] = await self._gather_with_limit(
            tasks, limit=self.PER_REQUEST_CONCURRENCY_LIMIT
        )
        return user_claim_norm, user_claim_core_norm, is_negated, list(all_results)

    def _load_config(self, config: dict[str, Any] | None):
        """Load config if there is any, use defaults if none."""
        limit = self.AGGREGATION_LIMIT
        use_non_factcheck = True

        if config is not None:
            limit = config.get("maxEvidence", limit)
            use_non_factcheck = config.get("useNonFactcheck", use_non_factcheck)
        return (limit, use_non_factcheck)

    def _sort_and_aggregate(
        self,
        results: list[ArticleResultModel],
        config: dict[str, Any] | None,
    ) -> list[ArticleResultModel]:
        """Sort by NLI presence → combined score → fact-check preference, then mark aggregated."""
        (limit, use_non_factcheck) = self._load_config(config)

        def sort_key(x):
            has_nli = 0 if x.nli_result else 1
            alignment_penalty = 0
            if x.verdict is not None and x.nli_result:
                nli = x.nli_result.relationship
                if nli == NLILabel.SUPPORT and x.verdict < -0.2:
                    alignment_penalty = 1
                elif nli == NLILabel.REFUTE and x.verdict > 0.35:
                    alignment_penalty = 1
            return (
                has_nli,
                alignment_penalty,
                -x.combined_relevance_score,
                0 if x.found_claim else 1,
            )

        results.sort(key=sort_key)

        aggregated: list[ArticleResultModel] = []
        for result in results:
            if len(aggregated) >= limit:
                break
            if not use_non_factcheck and result.found_verdict is None:
                continue
            aggregated.append(result)
        return aggregated

    # -----------------------------------------------------------------------
    # REST entry point
    # -----------------------------------------------------------------------

    async def verify_claim(
        self,
        user_claim: str,
        exclude_doc_ids: list[str] | None = None,
        exclude_articles: bool = False,
        config: dict[str, Any] | None = None,
    ) -> dict:
        async with self._semaphore:
            _, _, is_negated, all_results = await self._run_pipeline(
                user_claim,
                exclude_doc_ids=exclude_doc_ids,
                exclude_articles=exclude_articles,
            )

        skipped = [r for r in all_results if r.skip_reason]
        filtered = [r for r in all_results if not r.skip_reason]
        aggregated = self._sort_and_aggregate(filtered, config)
        stats = self.calculate_stats(aggregated)

        return {
            "skipped": skipped,
            "results": filtered,
            "overall_verdict": stats["overall_verdict"],
            "bias_divergence": stats["bias_divergence"],
            "truth_confidence_score": stats["truth_confidence_score"],
            "bias_consistency": stats["bias_consistency"],
            "is_negated": is_negated,
        }

    # -----------------------------------------------------------------------
    # WebSocket entry point
    # -----------------------------------------------------------------------

    async def verify_claim_stream_with_stats(
        self, user_claim: str, config: dict[str, Any] | None = None
    ):
        result_queue: asyncio.Queue = asyncio.Queue()

        async def _pipeline():
            try:
                async with self._semaphore:
                    _, _, _, tasks, search_hits = await self._prepare_pipeline_tasks(
                        user_claim
                    )
                    await result_queue.put(("hits", search_hits))

                    task_sem = asyncio.Semaphore(self.PER_REQUEST_CONCURRENCY_LIMIT)

                    async def _run_with_limit(coro):
                        async with task_sem:
                            return await coro

                    # Results are emitted to the queue as they complete — each one
                    # is forwarded to the client immediately by the consumer below.
                    for coro in asyncio.as_completed(
                        [_run_with_limit(t) for t in tasks]
                    ):
                        result = await coro
                        await result_queue.put(("result", result))
            except Exception as e:
                await result_queue.put(("error", e))
                return
            await result_queue.put(("done", None))

        pipeline_task = asyncio.create_task(_pipeline())

        results_raw: list[ArticleResultModel] = []

        while True:
            try:
                item = await asyncio.wait_for(result_queue.get(), timeout=15)
            except asyncio.TimeoutError:
                if pipeline_task.done():
                    break
                yield {"type": "heartbeat"}
                continue

            kind = item[0]

            if kind == "error":
                raise item[1]
            if kind == "hits":
                yield {"type": StreamEventType.SEARCH_HITS, "hits": item[1]}
            elif kind == "result":
                result: ArticleResultModel = item[1]
                results_raw.append(result)
                yield {
                    "type": StreamEventType.RESULT,
                    "data": result.model_dump(),
                    "skipped": bool(result.skip_reason),
                }
            elif kind == "done":
                break

        non_skipped = [r for r in results_raw if not r.skip_reason]
        aggregated = self._sort_and_aggregate(non_skipped, config)
        aggregated_ids = {r.doc_id for r in aggregated}

        final_stats = self.calculate_stats(aggregated)
        yield {
            "type": StreamEventType.STATS,
            "total_results": len(results_raw),
            "stats": final_stats,
            "doc_ids": list(aggregated_ids),
        }

        loop = asyncio.get_running_loop()
        remarks_inputs = [
            (
                result.chunk_texts or result.content,
                result.verdict if result.verdict is not None else 0.0,
                (
                    result.nli_result.relationship
                    if result.nli_result
                    else NLILabel.NEUTRAL
                ),
                result.source_type != "fact_check",
            )
            for result in aggregated
        ]
        all_remarks = await loop.run_in_executor(
            self.executor,
            self.remarks_generation_service.generate_remarks_batch,
            remarks_inputs,
        )
        for result, remarks in zip(aggregated, all_remarks):
            yield {
                "type": StreamEventType.REMARKS,
                "doc_id": result.doc_id,
                "remarks": remarks,
            }

        yield {"type": StreamEventType.COMPLETE}

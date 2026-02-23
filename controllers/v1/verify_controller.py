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
    ENTITY_GENERIC_TOKENS,
)
from dateparser.search import search_dates
from services import (
    EmbeddingService,
    EntityExtractionService,
    NLIService,
    RemarksGenerationService,
    StatsService,
)
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

        # Stricter thresholds for article filtering
        self.RELEVANCE_THRESHOLD = 0.3
        self.ENTITY_THRESHOLD = 0.4

        # Weights for combined relevance score (60% semantic, 40% entity)
        self.SEMANTIC_WEIGHT = 0.6
        self.ENTITY_WEIGHT = 0.4

        # Thread pool for CPU-intensive ML operations (2 workers optimal for PyTorch models)
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Limit deep analysis to top results to avoid excessive NLI + remarks tasks
        self.MAX_DEEP_ANALYSIS = 5

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize input text for consistent downstream processing.

        Steps:
        - Unicode normalization (NFKC)
        - Lowercasing
        - Removal of extra whitespace
        - Stripping leading/trailing whitespace

        Args:
            text (str): The input text to normalize.

        Returns:
            str: The normalized text.
        """
        if not isinstance(text, str):
            return ""

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Lowercase
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
        For each doc_id in doc_ids, finds the top-N most relevant article chunks using vector search.

        Args:
            embedding (list[float]): Embedding vector for the user claim.
            doc_ids (set[str]): Set of unique document IDs to search within.
            top_n (int, optional): Number of top chunks to retrieve per document. Defaults to 3.

        Returns:
            dict[str, list[tuple[ArticleChunk, float]]]: Mapping from doc_id to list of (chunk, similarity score) tuples.
        """
        chunks = self.db.find_similar_chunks_from_doc_ids(embedding, doc_ids)

        # Group by doc_id
        chunk_map: dict[str, list[tuple[ArticleChunk, float]]] = defaultdict(list)
        for chunk, distance in chunks:
            chunk_map[chunk.doc_id].append((chunk, 1 - distance))

        # For each doc_id, sort by distance and take top_n
        for doc_id, chunk_list in chunk_map.items():
            chunk_list.sort(key=lambda x: x[1])
            chunk_map[doc_id] = [
                (chunk, distance) for chunk, distance in chunk_list[:top_n]
            ]

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
        self, embedding: list[float], top_k: int = 20
    ) -> list[tuple[Claim, Article, float, str | None]]:
        """
        Retrieves the top_k most similar claims from the database using HNSW vector search,
        skipping claims with an 'UNKNOWN' verdict, and pairs each claim with its corresponding article.

        Args:
            embedding (list[float]): Embedding vector for the user claim.
            top_k (int, optional): Number of top similar claims to retrieve. Defaults to 20.

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
        Retrieves the top_k most relevant news articles, then grouping and ranking articles based on
        the highest similarity of their associated chunks. For each article, also provides the most
        relevant chunk text as context.

        Args:
            embedding (list[float]): Embedding vector for the user claim.
            top_k (int, optional): Number of top similar articles to retrieve. Defaults to 20.

        Returns:
            list[tuple[Article, float, str | None]]: List of tuples containing the article, similarity score, and relevant chunk text.
        """
        # 1: Get all similar chunks
        chunk_results = self.db.find_similar_chunks(embedding, top_k)

        # 2: Collect unique doc_ids from chunks
        unique_doc_ids = self.extract_unique_doc_ids(chunk_results)

        # 3: Get all articles related to chunks
        articles = self.db.find_articles_from_doc_ids(unique_doc_ids)

        # 4: Find MORE relevant chunks for each article (yield more chunks per article for better result)
        all_chunks_map = self.get_chunk_map(embedding, unique_doc_ids)

        # 5: build results using collected data
        results: list[tuple[Article, float, str | None]] = []

        for article in articles:
            article_relevant_chunks = all_chunks_map[article.doc_id]

            if len(article_relevant_chunks) == 0:
                results.append((article, 0.0, None))
                continue

            # Get first el of relevant chunk, then get its distance (index 1 of tuple)
            top_similarity_score = article_relevant_chunks[0][1]
            chunk_text = self.build_chunk_text(
                [chunk for chunk, _ in article_relevant_chunks]
            )
            results.append((article, top_similarity_score, chunk_text))

        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

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

        text_norm = self.normalize_text(text)
        article_title_norm = self.normalize_text(article_title)

        matches = 0.0
        total_weight = 0.0

        # Tokenize the normalized article content and title into sets of words
        text_tokens = set(re.findall(r"\b[a-zA-Z][a-zA-Z\-]{1,}\b", text_norm))
        title_tokens = set(
            re.findall(r"\b[a-zA-Z][a-zA-Z\-]{1,}\b", article_title_norm)
        )
        # Combine tokens from both content and title for comparison
        comparison_tokens = text_tokens.union(title_tokens)

        for entity in claim_entities:
            # Normalize the entity string
            entity_lower = self.normalize_text(entity)
            # Tokenize the entity into words
            entity_tokens_all = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{1,}\b", entity_lower)

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

            # Partial token match fallback for incomplete mentions (e.g., "Super Typhoon Uwan" vs "Uwan")
            if specific_entity_tokens and any(
                token in comparison_tokens for token in specific_entity_tokens
            ):
                matches += 0.7 * entity_weight

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
            entity_tokens = re.findall(
                r"\b[a-zA-Z][a-zA-Z\-]{1,}\b", self.normalize_text(entity)
            )
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
        text_norm = self.normalize_text(text)
        article_title_norm = self.normalize_text(article_title)
        comparison_tokens = set(re.findall(r"\b[a-zA-Z][a-zA-Z\-]{1,}\b", text_norm))
        comparison_tokens.update(
            re.findall(r"\b[a-zA-Z][a-zA-Z\-]{1,}\b", article_title_norm)
        )

        for entity in claim_entities:
            entity_tokens = re.findall(
                r"\b[a-zA-Z][a-zA-Z\-]{1,}\b", self.normalize_text(entity)
            )
            specific_tokens = [
                token
                for token in entity_tokens
                if token not in ENTITY_GENERIC_TOKENS and len(token) >= 3
            ]
            if any(token in comparison_tokens for token in specific_tokens):
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

    @staticmethod
    def compute_final_score(
        verdict: Verdict | None,
        source_bias: SourceBias | None,
        nli_label: NLILabel,
        nli_score: float,
        is_factcheck: bool = True,
        similarity_score: float = 0.0,
        article_content: str = "",
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

        Returns:
            float: The computed final score (signed and fuzzified, with magnitude reflecting strength).
        """
        if not is_factcheck:

            if nli_label == NLILabel.SUPPORT:
                base_score = 0.75
            elif nli_label == NLILabel.REFUTE:
                base_score = -0.75
            else:
                # Use entity match as a proxy for context relevance
                entity_match = 0.0
                if article_content and article_content.strip():
                    # If claim entities are found in article, treat as match (simulate previous boost)
                    # For simplicity, treat entity_match as 1.0 if similarity_score >= 0.4
                    entity_match = 1.0 if similarity_score >= 0.4 else 0.0

                if similarity_score >= 0.7:
                    base_score = min(0.8, similarity_score * 1.05)
                elif similarity_score >= 0.5:
                    base_score = similarity_score * 1.02
                elif entity_match and similarity_score >= 0.4:
                    base_score = min(0.6, similarity_score * 1.3)
                else:
                    base_score = similarity_score

            confidence_multiplier = 0.5 + (nli_score * 0.5)

            bias_weight = (
                SOURCE_BIAS_WEIGHT_MAP.get(source_bias, 0.7) if source_bias else 0.7
            )

            return round(base_score * confidence_multiplier * bias_weight, 2)

        if verdict is None or source_bias is None:
            return 0.0

        verdict_weight = VERDICT_WEIGHT_MAP.get(verdict, 0.5)
        bias_weight = SOURCE_BIAS_WEIGHT_MAP.get(source_bias, 0.7)
        nli_label_weight = NLI_LABEL_WEIGHT_MAP.get(nli_label, 0.5)

        return round(nli_score * bias_weight * verdict_weight * nli_label_weight, 2)

    async def verify_claim(self, user_claim: str, use_fallback: bool = True):
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

        Returns:
            dict: Dict containing lists of results, each containing article and claim details, NLI results, scores, and skip reasons.
        """

        user_claim_norm = self.normalize_text(user_claim)
        user_claim_for_matching = self.normalize_text(user_claim, lowercase=False)

        # Run embedding and entity extraction in parallel
        loop = asyncio.get_event_loop()
        claim_embedding, claim_entities = await asyncio.gather(
            loop.run_in_executor(
                self.executor,
                self.embedding_service.embed_text,
                user_claim_for_matching,
            ),
            loop.run_in_executor(
                self.executor, self.extract_entities, user_claim_for_matching
            ),
        )

        # Search for both FC and news articles
        factcheck_results = self.find_claims_with_articles(
            claim_embedding, self.MAX_DEEP_ANALYSIS
        )
        news_results = self.find_news_articles(claim_embedding, self.MAX_DEEP_ANALYSIS)

        # Process all results in parallel
        tasks = []

        # Process FC results
        for claim, article, similarity_score, chunk_texts in factcheck_results:
            tasks.append(
                self.process_result_async(
                    user_claim_norm=user_claim_norm,
                    claim_entities=claim_entities,
                    similarity_score=similarity_score,
                    article=article,
                    claim_text=claim.claim_text,
                    claim_verdict=claim.verdict,
                    source_bias=article.source_bias,
                    is_factcheck=True,
                    chunk_texts=chunk_texts,
                )
            )

        # Process news results
        for article, similarity_score, chunk_texts in news_results:
            if (article.type or "").strip().lower() == "fact-check":
                continue
            tasks.append(
                self.process_result_async(
                    user_claim_norm=user_claim_norm,
                    claim_entities=claim_entities,
                    similarity_score=similarity_score,
                    article=article,
                    claim_text=None,
                    claim_verdict=None,
                    source_bias=article.source_bias,
                    is_factcheck=False,
                    chunk_texts=chunk_texts,
                )
            )

        # Wait for all processing to complete
        results: list[ArticleResultModel] = await asyncio.gather(*tasks)

        # Filter and sort results
        skipped: list[ArticleResultModel] = []
        filtered_results: list[ArticleResultModel] = []

        for result in results:
            if len(result.skip_reason) > 0:
                skipped.append(result)
            else:
                filtered_results.append(result)

        filtered_results.sort(
            key=lambda x: (
                0 if x.nli_result else 1,
                0 if x.found_claim else 1,
                -x.combined_relevance_score,
            )
        )

        stats = self.stats_service.calculate_stats(filtered_results)

        return {
            "skipped": skipped,
            "results": filtered_results,
            "overall_verdict": stats["overall_verdict"],
            "bias_divergence": stats["bias_divergence"],
            "truth_confidence_score": stats["truth_confidence_score"],
            "bias_consistency": stats["bias_consistency"],
        }

    async def process_result_async(
        self,
        user_claim_norm: str,
        claim_entities: list[str],
        similarity_score: float,
        article: Article,
        claim_text: str | None,
        claim_verdict: str | None,
        source_bias: str,
        is_factcheck: bool,
        chunk_texts: str | None,
    ) -> ArticleResultModel:
        """
        Asynchronously processes a single claim-article or article result, computing entity match, relevance, NLI, verdict, and remarks.

        Args:
            user_claim_norm (str): Normalized user claim text.
            claim_entities (list[str]): List of entities extracted from the user claim.
            similarity_score (float): Semantic similarity score between user claim and article/claim.
            article (Article): The article object being evaluated.
            claim_text (str | None): The matched claim text, if available.
            claim_verdict (str | None): The verdict of the matched claim, if available.
            source_bias (str): The bias of the article's source.
            is_factcheck (bool): Whether the result is from a fact-checking source.
            chunk_texts (str | None): Relevant chunk text(s) for context.

        Returns:
            ArticleResultModel: Article and claim details, NLI results, scores, remarks, and skip reasons.
        """

        # For entity matching: use claim text for FC, or article content+title for news
        if is_factcheck and claim_text:
            entity_comparison_text = claim_text
            entity_comparison_title = ""  # Claim text is self-contained
        else:
            # For news articles, combine content and title
            entity_comparison_text = article.content or ""
            entity_comparison_title = article.title

        entity_match_score = self.calculate_entity_match_score(
            claim_entities, entity_comparison_text, entity_comparison_title
        )
        combined_relevance_score = (similarity_score * self.SEMANTIC_WEIGHT) + (
            entity_match_score * self.ENTITY_WEIGHT
        )

        # Truncate content ONLY for display
        display_content = (
            article.content[:500] + "..."
            if article.content and len(article.content) > 500
            else (article.content or "")
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
        }

        meets_relevance_gate = (
            similarity_score >= self.RELEVANCE_THRESHOLD
            and entity_match_score >= self.ENTITY_THRESHOLD
        )

        requires_specific_match = self.requires_specific_entity_match(claim_entities)
        has_specific_match = self.has_specific_entity_token_match(
            claim_entities, entity_comparison_text, entity_comparison_title
        )

        # Require direct claim-topic relevance: at least two claim keywords must appear in article content/title
        claim_keywords = [
            kw
            for kw in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", user_claim_norm)
            if kw not in ENTITY_GENERIC_TOKENS
        ]
        content_title = (entity_comparison_text + " " + entity_comparison_title).lower()
        keyword_matches = [
            kw.lower() for kw in claim_keywords if kw.lower() in content_title
        ]
        keyword_match = len(keyword_matches) >= 1  # Require at least one claim keyword

        if requires_specific_match and not has_specific_match:
            meets_relevance_gate = False
        if not keyword_match:
            meets_relevance_gate = False

        if meets_relevance_gate:
            # For NLI: use claim for FC, article content for news
            nli_text = (
                claim_text
                if is_factcheck and claim_text
                else (chunk_texts or article.content)
            )

            # Run NLI classification in thread pool
            loop = asyncio.get_event_loop()
            nli_label, nli_score, nli_avg = await loop.run_in_executor(
                self.executor,
                self.nli_service.classify_nli,
                user_claim_norm,
                nli_text,
            )

            result["nli_result"] = {
                "relationship": nli_label,
                "relationship_confidence": nli_score,
                "relationship_avg": nli_avg,
                "claim_source": article.source,
                "analyzed_text": self.truncate_at_sentence(nli_text, 200),
            }

            verdict_score = self.compute_final_score(
                verdict=Verdict(claim_verdict) if claim_verdict else None,
                source_bias=SourceBias(source_bias),
                nli_label=nli_label,
                nli_score=nli_score,
                is_factcheck=is_factcheck,
                similarity_score=similarity_score,
                article_content=article.content or "",
            )
            result["verdict"] = verdict_score
        else:
            if requires_specific_match and not has_specific_match:
                result["skip_reason"].append(
                    "Specific name/identifier from claim not found in matched result"
                )
            if similarity_score < self.RELEVANCE_THRESHOLD:
                result["skip_reason"].append(
                    f"Low semantic similarity ({similarity_score:.3f} < {self.RELEVANCE_THRESHOLD})"
                )
            if entity_match_score < self.ENTITY_THRESHOLD:
                result["skip_reason"].append(
                    f"Key entities not found ({entity_match_score:.3f} < {self.ENTITY_THRESHOLD})"
                )
            if not result["skip_reason"]:
                result["skip_reason"].append("Did not meet filtering criteria")

        return ArticleResultModel(**result)

    async def verify_claim_stream_with_stats(self, user_claim: str):
        """
        Streams search hits, then result items (filtered/skipped), then remarks updates.
        """
        user_claim_norm = self.normalize_text(user_claim)
        user_claim_for_matching = self.normalize_text(user_claim, lowercase=False)
        loop = asyncio.get_event_loop()

        # Embedding & entity extraction
        claim_embedding = await loop.run_in_executor(
            self.executor, self.embedding_service.embed_text, user_claim_for_matching
        )
        claim_entities = await loop.run_in_executor(
            self.executor, self.extract_entities, user_claim_for_matching
        )

        # 1. Stream search hits
        factcheck_results = self.find_claims_with_articles(claim_embedding)
        news_results = self.find_news_articles(claim_embedding)

        search_hits = []
        for claim, article, _, chunk_texts in factcheck_results:
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
        for article, _, chunk_texts in news_results:
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
                        user_claim_norm=user_claim_norm,
                        claim_entities=claim_entities,
                        similarity_score=similarity_score,
                        article=article,
                        claim_text=claim.claim_text,
                        claim_verdict=claim.verdict,
                        source_bias=article.source_bias,
                        is_factcheck=True,
                        chunk_texts=chunk_texts,
                    )
                )

        for article, similarity_score, chunk_texts in news_results:
            if article.doc_id not in processed_doc_ids:
                processed_doc_ids.add(article.doc_id)
                tasks.append(
                    self.process_result_async(
                        user_claim_norm=user_claim_norm,
                        claim_entities=claim_entities,
                        similarity_score=similarity_score,
                        article=article,
                        claim_text=None,
                        claim_verdict=None,
                        source_bias=article.source_bias,
                        is_factcheck=False,
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
        final_stats = self.stats_service.calculate_stats(results)
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

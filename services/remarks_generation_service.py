import os
from typing import Optional
import re

import torch
import langdetect
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    pipeline,
)

from services.translation_service import TranslationService


class RemarksGenerationService:
    def __init__(
        self,
        micro_llm_model: Optional[str] = None,
        use_llm: bool = True,
    ):

        self.translation_service = TranslationService()
        self.micro_llm_model_name = (
            micro_llm_model
            or os.getenv("MICRO_LLM_MODEL")
            or "Vamsi/T5_Paraphrase_Paws"
        )
        self.use_llm = use_llm

        self.micro_tokenizer = None
        self.micro_model = None
        self.bart_summarizer = None

        if self.use_llm:
            config = AutoConfig.from_pretrained(self.micro_llm_model_name)

            # Must be encoder-decoder model
            if not config.is_encoder_decoder:
                raise ValueError(
                    "Selected micro LLM must be an encoder-decoder (T5-based) model."
                )

            self.micro_tokenizer = AutoTokenizer.from_pretrained(
                self.micro_llm_model_name
            )
            self.micro_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.micro_llm_model_name
            )

            self.micro_model.eval()

            # BART summarizer fallback for broken excerpts
            try:
                self.bart_summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if torch.cuda.is_available() else -1,
                )
            except Exception:
                pass

    def _verdict_score_meaning(
        self, verdict_score: float, nli_relationship: str = "neutral"
    ) -> str:
        relationship = (nli_relationship or "neutral").lower()

        # Neutral: softer language
        if relationship == "neutral":
            if verdict_score < 0:
                return "is inconclusive or slightly refuted"
            if verdict_score < 0.33:
                return "is inconclusive"
            if verdict_score < 0.66:
                return "is inconclusive or slightly supported"
            return "is inconclusive or mostly supported"

        # Align verdict sign with NLI relationship
        if relationship in {"refute", "contradiction"}:
            verdict_score = -abs(verdict_score)
        elif relationship in {"support", "entailment"}:
            verdict_score = abs(verdict_score)

        # Map score to category
        if verdict_score <= -0.66:
            return "is strongly refuted"
        if verdict_score <= -0.33:
            return "is mostly refuted"
        if verdict_score < 0.33:
            return "is neither clearly supported nor refuted"
        if verdict_score < 0.66:
            return "is mostly supported"
        return "is strongly supported"

    @staticmethod
    def _is_nonsensical_excerpt(text: str) -> bool:
        """Detect truncated or incomplete excerpts"""
        if not text:
            return True

        text_stripped = text.strip()

        # Article template metadata
        if "IF YOUR TIME IS SHORT" in text:
            return True

        # Starts with rating label (fact-check metadata)
        if re.match(
            r"^(True|False|Mostly\s+True|Mostly\s+False|Mixed|No\s+Evidence|Needs\s+Context|Partially\s+True|Partially\s+False)\s*:?\s*",
            text_stripped,
            re.IGNORECASE,
        ):
            return True

        # Too short unless ends with punctuation
        if len(text_stripped) < 20 and not re.search(r"[.!?]$", text_stripped):
            return True

        # Only title suffix, nothing else
        if re.match(
            r"^[A-Za-z\s\.]+\s+(?:Gov|Rep|Sen|Dr|Mr|Ms|Mrs|Inc|Ltd|Co|Corp)\.?$",
            text_stripped,
            re.I,
        ):
            # Exclude if many words (full clause)
            word_count = len(text_stripped.split())
            if word_count <= 3:
                return True

        # Ends with title suffix
        if re.search(
            r"\b(?:[A-Z]\.?){1,3}\s+(?:Gov|Rep|Sen|Dr|Mr|Ms|Mrs|Inc|Ltd|Co|Corp)\.?\s*$",
            text_stripped,
        ):
            return True

        # Dangling preposition/conjunction
        if re.search(
            r"\b(?:and|or|in|of|to|at|on|for|with|by|from|as|but|nor|the|a|an|it)\s*$",
            text_stripped,
            re.I,
        ):
            return True

        # Short incomplete fragment
        if len(text_stripped) < 50 and re.search(
            r"\b(?:in|of|to|at|on|for|by|from|as|with)\s+(?:it|a|an|the|that|this|one|some|all|by|for|to)\s*$",
            text_stripped,
            re.I,
        ):
            return True

        # Ellipsis truncation
        if text_stripped.endswith("..."):
            return True

        # Incomplete date/month reference
        if re.search(
            r"\b(?:on|in|at|since|until|from|during)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\.?\s*$",
            text_stripped,
            re.I,
        ):
            return True

        # Only initials
        if re.match(r"^[A-Z]\.?\s+[A-Z]\.?\s*$", text_stripped):
            return True

        # Open bracket/quote
        if re.search(r"[\(\[\{]$", text_stripped):
            return True

        return False

    def _generate_remarks_from_full_text(
        self, full_text: str, verdict_score: float, nli_relationship: str = "neutral"
    ) -> str:
        """BART fallback for nonsensical excerpts"""
        meaning = self._verdict_score_meaning(verdict_score, nli_relationship)

        if not self.bart_summarizer or not full_text or len(full_text) < 50:
            return (
                f"The claim {meaning}, "
                f"with a verdict score of {verdict_score:.2f} on a scale from -1 to 1."
            )

        try:
            # Token limit
            max_input = min(len(full_text), 1024)
            input_text = full_text[:max_input]

            # Dynamic length (40-60% of input)
            input_word_count = len(input_text.split())
            max_length = max(15, int(input_word_count * 0.5))
            min_length = min(10, max(5, int(input_word_count * 0.25)))

            summary = self.bart_summarizer(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )

            summary_text = summary[0]["summary_text"] if summary else ""
            if summary_text:
                summary_text = self._normalize_leading_determiner(summary_text)
                summary_text = self._ensure_sentence_end(summary_text)
                return (
                    f"The article reports that {summary_text} "
                    f"Based on this evidence, the claim {meaning}, "
                    f"with a verdict score of {verdict_score:.2f} on a scale from -1 to 1."
                )
        except Exception:
            pass

        return (
            f"The claim {meaning}, "
            f"with a verdict score of {verdict_score:.2f} on a scale from -1 to 1."
        )

    @staticmethod
    def _normalize_leading_determiner(text: str) -> str:
        """Lower-case leading determiners"""
        return re.sub(
            r"^(a|an|the|there|this|that|these|those|according)\b",
            lambda m: m.group(1).lower(),
            text,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _ensure_sentence_end(text: str) -> str:
        """Add sentence punctuation if missing"""
        if not text:
            return ""

        stripped = text.rstrip()

        # Remove double punctuation after quote
        stripped = re.sub(r'([.!?]["\'])\.$', r"\1", stripped)

        # Already has punctuation
        if re.search(r'[.!?]["\']?$', stripped):
            return stripped

        # Add period before quote
        if re.search(r'["\'][\s]*$', stripped):
            stripped = re.sub(r'(["\'][\s]*)$', r".\1", stripped)
            return stripped

        return stripped + "."

    @staticmethod
    def _extract_excerpt(text: str, max_chars: int = 240) -> str:

        if not text:
            return ""

        cleaned = re.sub(r"\s+", " ", text.strip())
        cleaned = re.sub(r"(?:\.\.\.|…)$", "", cleaned).strip()

        # Remove common news article prefixes
        cleaned = re.sub(r"^\(UPDATED\)\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"^[A-Z][A-Za-z\s\.'-]+,\s*[A-Z][A-Za-z\s\.'-]+\s*—\s*",
            "",
            cleaned,
        )

        # Remove fact-check metadata
        cleaned = re.sub(r"^Claim:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^Rating:\s*", "", cleaned, flags=re.IGNORECASE)

        # Remove fact-check rating labels (True, False, Mostly True, etc.)
        cleaned = re.sub(
            r"^(True|False|Mostly\s+True|Mostly\s+False|Mixed|No\s+Evidence|Needs\s+Context|Partially\s+True|Partially\s+False)\s*:?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # Normalize whitespace after removals
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

        if len(cleaned) <= max_chars:
            return cleaned

        # Find complete sentence within limit
        truncated = cleaned[:max_chars]
        sentence_matches = list(re.finditer(r"[.!?](?=\s+[A-Z0-9]|$)", truncated))

        if sentence_matches:
            last_end = sentence_matches[-1].end()
            return cleaned[:last_end].strip()

        # Extend search +100 chars
        extended_limit = min(len(cleaned), max_chars + 100)
        extended = cleaned[:extended_limit]
        sentence_matches = list(re.finditer(r"[.!?](?=\s+[A-Z0-9]|$)", extended))

        if sentence_matches:
            last_end = sentence_matches[-1].end()
            return cleaned[:last_end].strip()

        # Truncate at word boundary
        snippet = truncated.rsplit(" ", 1)[0].rstrip() + "..."
        return snippet

    @staticmethod
    def _clean_excerpt(excerpt: str) -> str:

        if not excerpt:
            return ""
        excerpt = re.sub(r"[\s\-–—:;,]+$|[\s\(\[\{]+$|[\"']+$", "", excerpt)
        if excerpt.count("(") > excerpt.count(")"):
            excerpt = excerpt.rsplit("(", 1)[0].rstrip()
        if excerpt.count('"') % 2 == 1 or excerpt.count("“") > excerpt.count("”"):
            excerpt = excerpt.rsplit('"', 1)[0].rstrip()
            excerpt = excerpt.rsplit("“", 1)[0].rstrip()

        for pat in (
            r"\s+before\s+[^.!?]*\bexpected\b$",
            r"\s+as\s+[^.!?]*\btowards?\b$",
            r"\s+heads\s+towards?\b$",
            r"\s+(during|after|before)\s+[A-Z][A-Za-z\-]*$",
        ):
            excerpt = re.sub(pat, "", excerpt, flags=re.I)

        org_tail = r"\bthe\s+(National|Department|Ministry|Bureau|Agency|Administration|Council|Commission|Committee|Office|Center|Authority|Organization)\b\s*$"
        if re.search(org_tail, excerpt, re.I):
            excerpt = excerpt.rsplit(",", 1)[0].rstrip()

        stop_tail = r"\b(the|a|an|and|or|in|of|to|at|on|for|with|by|from|as|into|during|after|before|towards?|according|expected)\b\s*$"
        if re.search(stop_tail, excerpt, re.I):
            excerpt = (
                excerpt.rsplit(",", 1)[0].rstrip()
                if "," in excerpt
                else re.sub(r"\s+\w+$", "", excerpt).rstrip()
            )

        excerpt = re.sub(r"\b(international name)\s*$", "", excerpt, flags=re.I)
        excerpt = re.sub(r"([.!?][\"”'])\.+$", r"\1", excerpt)
        return re.sub(r"\s{2,}", " ", excerpt).strip()

    def _paraphrase_excerpt(self, excerpt: str) -> str:

        if (
            not self.use_llm
            or not self.micro_model
            or not self.micro_tokenizer
            or len(excerpt) < 80
        ):
            return excerpt

        excerpt_text = excerpt.rstrip(".")

        if any(
            re.search(p, excerpt_text, re.I)
            for p in (
                r"\b(?:Mon|Tue|Tues|Wed|Thu|Thur|Fri|Sat|Sun)(?:day)?\b",
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\b",
            )
        ):
            return excerpt

        if re.search(r"[\(\"']$", excerpt_text):
            return excerpt

        if re.search(
            r"\b(the|a|an|and|or|in|of|to|at|on|for|with|by|from|as|into|during|after|before|towards?|according)\s*$",
            excerpt_text,
            re.I,
        ):
            return excerpt

        if re.search(r"[\d,]+\s+(individuals?|persons?|people)$", excerpt_text, re.I):
            return excerpt

        prompt = f"paraphrase: {excerpt} </s>"

        try:
            inputs = self.micro_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )

            with torch.no_grad():
                outputs = self.micro_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )

            decoded = self.micro_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            paraphrased = decoded.strip() or excerpt

            # Clean up spacing artifacts from model output
            paraphrased = re.sub(r"\(\s+", "(", paraphrased)
            paraphrased = re.sub(r"\s+\)", ")", paraphrased)
            paraphrased = re.sub(r"\s+([,.;:!?])", r"\1", paraphrased)
            paraphrased = re.sub(r"\s{2,}", " ", paraphrased).strip()

            # Reject if too short or ends with verb
            if len(paraphrased) < max(40, int(len(excerpt) * 0.6)) or re.search(
                r"\b(is|are|was|were)\.?$", paraphrased, re.IGNORECASE
            ):
                return excerpt

            return paraphrased

        except Exception:
            return excerpt

    def _format_remarks_with_article(
        self, excerpt: str, meaning: str, verdict_score: float
    ) -> str:
        """Format remarks with article context"""
        return (
            f"The article reports that {excerpt} "
            f"Based on this evidence, the claim {meaning}, "
            f"with a verdict score of {verdict_score:.2f} on a scale from -1 to 1."
        )

    def generate_remarks(
        self,
        input_text: str,
        verdict_score: float,
        nli_relationship: str = "neutral",
        use_llm: bool = True,
    ) -> str:

        # Detect language and translate
        try:
            lang = langdetect.detect(input_text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "unknown"

        allow_paraphrase = True
        if lang != "en":
            input_text = self.translation_service.translate_to_english(input_text)
            allow_paraphrase = False

        excerpt = self._clean_excerpt(self._extract_excerpt(input_text))
        meaning = self._verdict_score_meaning(verdict_score, nli_relationship)

        # Use BART if excerpt is broken
        if self._is_nonsensical_excerpt(excerpt):
            if use_llm and allow_paraphrase:
                return self._generate_remarks_from_full_text(
                    input_text, verdict_score, nli_relationship
                )
            else:
                # Try BART fallback even if use_llm is False
                bart_result = self._generate_remarks_from_full_text(
                    input_text, verdict_score, nli_relationship
                )
                if "The article reports that" in bart_result:
                    return bart_result
                # Only return no-context remark if BART also failed
                return (
                    f"The claim {meaning}, "
                    f"with a verdict score of {verdict_score:.2f} on a scale from -1 to 1."
                )

        if not excerpt:
            # Try BART fallback for empty excerpt
            if use_llm and allow_paraphrase:
                bart_result = self._generate_remarks_from_full_text(
                    input_text, verdict_score, nli_relationship
                )
                if "The article reports that" in bart_result:
                    return bart_result
            return (
                f"The claim {meaning}, "
                f"with a verdict score of {verdict_score:.2f} on a scale from -1 to 1."
            )

        # Normalize determiners
        if re.match(r"^(A|An|The|There|This|That|These|Those)\s", excerpt):
            excerpt = excerpt[0].lower() + excerpt[1:]
        if not re.search(r"[.!?]$", excerpt):
            excerpt = excerpt.rstrip() + "."

        # Apply T5 paraphrasing
        if use_llm and allow_paraphrase:
            paraphrased_excerpt = self._paraphrase_excerpt(excerpt)
            paraphrased_excerpt = self._normalize_leading_determiner(
                paraphrased_excerpt
            )
            paraphrased_excerpt = self._ensure_sentence_end(paraphrased_excerpt)
            return self._format_remarks_with_article(
                paraphrased_excerpt, meaning, verdict_score
            )

        return self._format_remarks_with_article(excerpt, meaning, verdict_score)

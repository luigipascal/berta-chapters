"""
Production-quality text preprocessing module for Chapter 10: NLP Basics.
Provides tokenization, stopword removal, lemmatization, vocabulary building,
and a reusable TextPreprocessor class.
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional NLTK imports (lazy to avoid import errors if not installed)
def _get_nltk():
    try:
        import nltk
        return nltk
    except ImportError:
        raise ImportError("nltk is required. Install with: pip install nltk") from None


def tokenize_text(text: str, method: str = "word") -> List[str]:
    """
    Tokenize text into sentences or words.

    Args:
        text: Input string to tokenize.
        method: 'word' for word-level tokens, 'sentence' for sentence-level.

    Returns:
        List of tokens (words or sentences).

    Raises:
        ValueError: If method is not 'word' or 'sentence'.
    """
    if not text or not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    nltk = _get_nltk()
    try:
        if method == "word":
            return nltk.word_tokenize(text)
        if method == "sentence":
            return nltk.sent_tokenize(text)
    except LookupError:
        logger.warning("NLTK tokenizer data missing. Run: nltk.download('punkt')")
        if method == "word":
            return re.findall(r"\b\w+\b", text)
        if method == "sentence":
            return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    raise ValueError("method must be 'word' or 'sentence'")


def remove_stopwords(
    tokens: List[str],
    language: str = "english",
    extra_stopwords: Optional[List[str]] = None,
) -> List[str]:
    """
    Remove common words that don't carry meaning.

    Args:
        tokens: List of word tokens.
        language: Language for NLTK stopwords (e.g. 'english').
        extra_stopwords: Additional words to remove.

    Returns:
        List of tokens with stopwords removed.
    """
    if not tokens:
        return []
    nltk = _get_nltk()
    try:
        stop = set(nltk.corpus.stopwords.words(language))
    except LookupError:
        logger.warning("NLTK stopwords missing. Run: nltk.download('stopwords')")
        stop = set()
    if extra_stopwords:
        stop.update(w.lower() for w in extra_stopwords)
    return [t for t in tokens if t.lower() not in stop]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Convert words to base form (lemmas) using NLTK WordNetLemmatizer.
    Expects tokens in lowercase for better lemmatization.

    Args:
        tokens: List of word tokens.

    Returns:
        List of lemmatized tokens.
    """
    if not tokens:
        return []
    nltk = _get_nltk()
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
    except LookupError:
        logger.warning("NLTK wordnet missing. Run: nltk.download('wordnet')")
        return list(tokens)
    return [lemmatizer.lemmatize(t.lower()) for t in tokens]


def clean_text(
    text: str,
    remove_stop: bool = True,
    lemmatize: bool = True,
    min_length: int = 1,
) -> str:
    """
    Full preprocessing pipeline in one function: tokenize → optional stopwords
    → optional lemmatization → rejoin.

    Args:
        text: Raw input text.
        remove_stop: Whether to remove stopwords.
        lemmatize: Whether to lemmatize.
        min_length: Minimum token length to keep.

    Returns:
        Cleaned text as a single string (space-joined tokens).
    """
    if not text or not isinstance(text, str):
        return ""
    tokens = tokenize_text(text, method="word")
    tokens = [t for t in tokens if len(t) >= min_length]
    if remove_stop:
        tokens = remove_stopwords(tokens)
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def build_vocabulary(
    texts: List[str],
    min_frequency: int = 2,
    max_size: Optional[int] = None,
) -> Dict[str, int]:
    """
    Create a vocabulary from a list of texts: word -> index (1-based; 0 reserved for OOV).

    Args:
        texts: List of raw or preprocessed text strings.
        min_frequency: Minimum occurrence count for a word to be included.
        max_size: Maximum vocabulary size (most frequent words kept). None = no limit.

    Returns:
        Dictionary mapping word to integer index. Index 0 is reserved for unknown/OOV.
    """
    from collections import Counter
    counter: Counter = Counter()
    for text in texts:
        if not text:
            continue
        tokens = tokenize_text(text, method="word")
        tokens = [t.lower() for t in tokens if len(t) >= 1]
        counter.update(tokens)
    # Filter by min_frequency
    vocab_list = [w for w, c in counter.most_common() if c >= min_frequency]
    if max_size is not None:
        vocab_list = vocab_list[: max_size]
    # 0 reserved for OOV
    return {w: i + 1 for i, w in enumerate(vocab_list)}


class TextPreprocessor:
    """
    Full preprocessing pipeline as a class. Can fit vocabulary from texts
    and transform new texts consistently.
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_token_length: int = 1,
    ):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length
        self.vocabulary_: Optional[Dict[str, int]] = None

    def fit(self, texts: List[str]) -> "TextPreprocessor":
        """
        Learn vocabulary from texts (optional; used for consistent indexing).
        Preprocessing options are applied before building vocab.
        """
        cleaned = []
        for text in texts:
            t = clean_text(
                text,
                remove_stop=self.remove_stopwords,
                lemmatize=self.lemmatize,
                min_length=self.min_token_length,
            )
            cleaned.append(t)
        self.vocabulary_ = build_vocabulary(cleaned, min_frequency=1)
        return self

    def transform(self, texts: List[str]) -> List[str]:
        """
        Apply preprocessing to texts. Does not update vocabulary.
        """
        return [
            clean_text(
                t,
                remove_stop=self.remove_stopwords,
                lemmatize=self.lemmatize,
                min_length=self.min_token_length,
            )
            for t in texts
        ]

    def fit_transform(self, texts: List[str]) -> List[str]:
        """Fit vocabulary from texts and return preprocessed text list."""
        self.fit(texts)
        return self.transform(texts)

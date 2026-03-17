"""
Word embedding utilities for Chapter 10: NLP Basics.
Load pre-trained embeddings (e.g. GloVe), similarity, analogies, and efficient lookup.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_pretrained_embeddings(
    model_name: str = "glove",
    path: Optional[str] = None,
    dim: int = 100,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Load pre-trained word embeddings from a file.

    Expects a text file with lines: "word v1 v2 ... vd" (space-separated).
    If path is None, returns empty dict and zero array (caller can load from URL/local file).

    Args:
        model_name: Label for logging (e.g. 'glove').
        path: Path to embedding file. If None, returns empty structures.
        dim: Expected embedding dimension (used for validation).

    Returns:
        (word2vec_dict, vectors_array) where vectors_array is (V, dim) for known words.
    """
    word2vec: Dict[str, np.ndarray] = {}
    vectors_list: List[np.ndarray] = []
    if not path or not Path(path).exists():
        logger.warning("No embedding path provided or file not found. Returning empty embeddings.")
        return word2vec, np.zeros((0, dim), dtype=np.float32)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            try:
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            except ValueError:
                continue
            if len(vec) != dim:
                continue
            word2vec[word] = vec
            vectors_list.append(vec)

    if not vectors_list:
        return word2vec, np.zeros((0, dim), dtype=np.float32)
    return word2vec, np.stack(vectors_list)


def get_word_embedding(
    word: str,
    embeddings: Dict[str, np.ndarray],
    default: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Get embedding for a single word. Tries lowercased form if not found.

    Args:
        word: Word to look up.
        embeddings: Dictionary of word -> vector.
        default: Optional vector to return for OOV (e.g. zeros).

    Returns:
        Embedding array or default/None if not found.
    """
    v = embeddings.get(word)
    if v is not None:
        return v
    v = embeddings.get(word.lower())
    if v is not None:
        return v
    return default


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    Returns value in [-1, 1]. Handles zero vectors.
    """
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (n1 * n2))


def find_similar_words(
    word: str,
    embeddings: Dict[str, np.ndarray],
    n: int = 5,
    exclude_self: bool = True,
) -> List[Tuple[str, float]]:
    """
    Find most similar words to a given word by cosine similarity.

    Args:
        word: Query word.
        embeddings: Dictionary of word -> vector.
        n: Number of similar words to return.
        exclude_self: If True, exclude the query word from results.

    Returns:
        List of (word, similarity) tuples, sorted by similarity descending.
    """
    v = get_word_embedding(word, embeddings)
    if v is None:
        return []
    results: List[Tuple[str, float]] = []
    for w, vec in embeddings.items():
        if exclude_self and w == word:
            continue
        sim = cosine_similarity(v, vec)
        results.append((w, sim))
    results.sort(key=lambda x: -x[1])
    return results[:n]


def word_analogy(
    word1: str,
    word2: str,
    word3: str,
    embeddings: Dict[str, np.ndarray],
    n: int = 5,
) -> List[Tuple[str, float]]:
    """
    Solve word analogies: word1 is to word2 as word3 is to ?
    Uses vector offset: target ≈ vec(word3) + vec(word2) - vec(word1).
    Returns top-n candidates by cosine similarity to target.
    """
    v1 = get_word_embedding(word1, embeddings)
    v2 = get_word_embedding(word2, embeddings)
    v3 = get_word_embedding(word3, embeddings)
    if v1 is None or v2 is None or v3 is None:
        return []
    target = v3 + v2 - v1
    results: List[Tuple[str, float]] = []
    exclude = {word1, word2, word3}
    for w, vec in embeddings.items():
        if w in exclude:
            continue
        sim = cosine_similarity(target, vec)
        results.append((w, sim))
    results.sort(key=lambda x: -x[1])
    return results[:n]


class EmbeddingIndex:
    """
    Efficient nearest-neighbor search in embedding space using brute-force
    (for small-to-medium vocabs). For very large vocabs, consider FAISS or Annoy.
    """

    def __init__(self, embeddings: Dict[str, np.ndarray]):
        self.words = list(embeddings.keys())
        self.vectors = np.stack([embeddings[w] for w in self.words]).astype(np.float32)
        self._norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self._norms[self._norms == 0] = 1.0

    def most_similar(self, word: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar words efficiently via cosine similarity (normalized dot product).
        """
        idx = None
        for i, w in enumerate(self.words):
            if w == word:
                idx = i
                break
        if idx is None:
            return []
        q = self.vectors[idx : idx + 1]
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        sims = np.dot(self.vectors, q.T).ravel() / (self._norms.ravel() * qn)
        sims[idx] = -2.0  # exclude self
        top = np.argsort(-sims)[:n]
        return [(self.words[i], float(sims[i])) for i in top]

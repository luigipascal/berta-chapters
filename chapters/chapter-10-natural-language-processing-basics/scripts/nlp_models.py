"""
Pre-built NLP models for common tasks: sentiment, text classification, NER, similarity.
Uses scikit-learn, optional spaCy, and optional Keras/TensorFlow.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Train and use sentiment analysis models (e.g. logistic regression on TF-IDF).
    """

    def __init__(self, model_type: str = "logistic"):
        self.model_type = model_type
        self.vectorizer: Any = None
        self.model: Any = None
        self._sklearn = None

    def _get_sklearn(self):
        if self._sklearn is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                self._sklearn = (TfidfVectorizer, LogisticRegression)
            except ImportError as e:
                raise ImportError("scikit-learn is required for SentimentAnalyzer") from e
        return self._sklearn

    def train(
        self,
        texts: List[str],
        labels: List[int],
        **kwargs: Any,
    ) -> "SentimentAnalyzer":
        """Train the sentiment model on (texts, labels). Labels: 0 = negative, 1 = positive."""
        TfidfVectorizer, LogisticRegression = self._get_sklearn()
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), **kwargs)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=500, random_state=42)
        self.model.fit(X, labels)
        return self

    def predict(self, text: str) -> Tuple[int, float]:
        """
        Return (predicted class, confidence/probability of positive class).
        """
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        pred = int(self.model.predict(X)[0])
        pos_idx = 1 if 1 in self.model.classes_ else 0
        conf = float(proba[pos_idx])
        return pred, conf


class TextClassifier:
    """
    Multi-class text classification with optional LSTM (Keras).
    Falls back to TF-IDF + logistic regression if LSTM not used.
    """

    def __init__(self, num_classes: int, model_type: str = "logistic"):
        self.num_classes = num_classes
        self.model_type = model_type
        self.vectorizer: Any = None
        self.model: Any = None
        self.classes_: Optional[np.ndarray] = None

    def train(
        self,
        texts: List[str],
        labels: List[int],
        epochs: int = 10,
        **kwargs: Any,
    ) -> "TextClassifier":
        """Train classifier. For 'logistic', uses TF-IDF + LogisticRegression."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except ImportError as e:
            raise ImportError("scikit-learn required for TextClassifier") from e
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=500, random_state=42)
        self.model.fit(X, labels)
        self.classes_ = np.array(self.model.classes_)
        return self

    def predict(self, text: str) -> Tuple[int, np.ndarray]:
        """Return (predicted class index, probability array)."""
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        pred = int(self.model.predict(X)[0])
        return pred, proba

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get probabilities for all classes for each text."""
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)


class NERModel:
    """
    Named Entity Recognition using spaCy. Extract entities and optionally visualize.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self._load()

    def _load(self) -> None:
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
        except OSError:
            logger.warning("spaCy model '%s' not found. Run: python -m spacy download %s", self.model_name, self.model_name)
            self.nlp = None

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities and their types. Returns list of dicts with
        'text', 'label', 'start', 'end'.
        """
        if self.nlp is None:
            return []
        doc = self.nlp(text)
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

    def visualize_entities(self, text: str) -> None:
        """Display entities with colors (requires spaCy and running in env that supports displacy)."""
        if self.nlp is None:
            logger.warning("spaCy not loaded; cannot visualize.")
            return
        try:
            from spacy import displacy
            doc = self.nlp(text)
            displacy.render(doc, style="ent", jupyter=True)
        except Exception as e:
            logger.warning("displacy render failed: %s", e)


class TextSimilarity:
    """
    Find and cluster similar texts using TF-IDF or cosine similarity.
    """

    def __init__(self, method: str = "tfidf"):
        self.method = method
        self.vectorizer: Any = None
        self.matrix: Optional[np.ndarray] = None

    def _get_sklearn(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
            return TfidfVectorizer, sk_cosine
        except ImportError as e:
            raise ImportError("scikit-learn required for TextSimilarity") from e

    def fit(self, texts: List[str]) -> "TextSimilarity":
        """Fit vectorizer and store matrix for later similarity/clustering."""
        TfidfVectorizer, _ = self._get_sklearn()
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.matrix = self.vectorizer.fit_transform(texts)
        return self

    def similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts (cosine on TF-IDF vectors)."""
        _, sk_cosine = self._get_sklearn()
        if self.vectorizer is None:
            self.fit([text1, text2])
        X = self.vectorizer.transform([text1, text2])
        return float(sk_cosine(X[0:1], X[1:2])[0, 0])

    def cluster_texts(
        self,
        texts: List[str],
        n_clusters: int = 3,
    ) -> np.ndarray:
        """Cluster texts based on TF-IDF similarity. Returns cluster labels."""
        try:
            from sklearn.cluster import KMeans
        except ImportError as e:
            raise ImportError("scikit-learn required for cluster_texts") from e
        if self.matrix is None:
            self.fit(texts)
        X = self.matrix
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(X)

import numpy as np
import tensorflow as tf
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Any


class IntentClassifier:
    def __init__(self, train_dir: str, batch_size: int = 16, seed: int = 42):
        self.batch_size = batch_size
        self.seed = seed
        self.train_dir = train_dir
        self.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=500)
        self.train_texts: List[str] = []
        self.train_labels: List[int] = []
        self.intent_names: List[str] = []
        self.is_trained = False

    def load_data(self) -> None:
        raw_train_ds = tf.keras.utils.text_dataset_from_directory(
            directory=self.train_dir,
            batch_size=self.batch_size,
            validation_split=0.2,
            seed=self.seed,
            subset='training',
            shuffle=True
        )

        self.intent_names = raw_train_ds.class_names

        for text_batch, label_batch in raw_train_ds:
            decoded_texts = [text.numpy().decode('utf-8') for text in text_batch]
            self.train_texts.extend(decoded_texts)
            self.train_labels.extend(label_batch.numpy())

    def train(self) -> None:
        if not self.train_texts:
            raise ValueError("No training data loaded. Call load_data() first.")

        X_train = self.vectorizer.fit_transform(self.train_texts)
        self.model.fit(X_train, self.train_labels)
        self.is_trained = True

    def calculate_similarity(self, query: str, texts: List[str], top_k: int = 5):
        query_vec = self.vectorizer.transform([query]).toarray()[0]
        text_vecs = self.vectorizer.transform(texts).toarray()

        similarities = []
        for text_vec in text_vecs:
            cos_sim = np.dot(query_vec, text_vec) / (norm(query_vec) * norm(text_vec))
            similarities.append(cos_sim)

        return np.argsort(similarities)[-top_k:][::-1], np.sort(similarities)[-top_k:][::-1]

    def predict(self, text: str, threshold: float = 0.3) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        example_vectorized = self.vectorizer.transform([text])
        predicted_label = self.model.predict(example_vectorized)[0]
        confidence = self.model.predict_proba(example_vectorized).max()

        similar_indices, similarities = self.calculate_similarity(text, self.train_texts)

        result = {
            'predicted_intent': self.intent_names[predicted_label],
            'confidence': float(confidence),  # Convert numpy float to Python float
            'similar_examples': []
        }

        for idx, sim_score in zip(similar_indices, similarities):
            if sim_score > threshold:
                result['similar_examples'].append({
                    'text': self.train_texts[idx],
                    'intent': self.intent_names[self.train_labels[idx]],
                    'similarity': float(sim_score)  # Convert numpy float to Python float
                })

        return result


# Singleton instance for the API
_classifier_instance = None


def get_classifier(train_dir: str) -> IntentClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier(train_dir)
        _classifier_instance.load_data()
        _classifier_instance.train()
    return _classifier_instance

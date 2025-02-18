from collections import Counter
from typing import Dict, List, Any, Tuple

import numpy as np
import tensorflow as tf
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


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

    def calculate_similarity(self, query: str, texts: List[str], top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        query_vec = self.vectorizer.transform([query]).toarray()[0]
        text_vecs = self.vectorizer.transform(texts).toarray()

        similarities = []
        for text_vec in text_vecs:
            cos_sim = np.dot(query_vec, text_vec) / (norm(query_vec) * norm(text_vec))
            similarities.append(cos_sim)

        # Ritorna gli indici dei top_k più simili e i loro punteggi
        return np.argsort(similarities)[-top_k:][::-1], np.sort(similarities)[-top_k:][::-1]

    def get_corrected_intent(self, similar_examples: List[Dict],
                             predicted_intent: str,
                             similarity_threshold: float = 0.7,
                             consensus_threshold: float = 0.8) -> Tuple[str, bool]:

        # Filtra solo gli esempi con alta similarità
        high_similarity_examples = [
            example for example in similar_examples
            if example['similarity'] > similarity_threshold
        ]

        if not high_similarity_examples:
            return predicted_intent, False

        # Conta gli intent degli esempi più simili
        intent_counts = Counter(
            example['intent'] for example in high_similarity_examples
        )

        if not intent_counts:
            return predicted_intent, False

        # Trova l'intent più comune tra gli esempi simili
        most_common_intent, count = intent_counts.most_common(1)[0]
        consensus_ratio = count / len(high_similarity_examples)

        # Se c'è un forte consenso per un intent diverso, correggi la predizione
        if (consensus_ratio >= consensus_threshold and
                most_common_intent != predicted_intent):
            return most_common_intent, True

        return predicted_intent, False

    def predict(self, text: str,
                similarity_threshold: float = 0.7,
                consensus_threshold: float = 0.8) -> Dict[str, Any]:

        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Predizione iniziale con regressione logistica
        example_vectorized = self.vectorizer.transform([text])
        predicted_label = self.model.predict(example_vectorized)[0]
        confidence = self.model.predict_proba(example_vectorized).max()
        initial_intent = self.intent_names[predicted_label]

        # Calcolo similarità con esempi di training
        similar_indices, similarities = self.calculate_similarity(text, self.train_texts)

        # Prepara lista di esempi simili
        similar_examples = []
        for idx, sim_score in zip(similar_indices, similarities):
            similar_examples.append({
                'text': self.train_texts[idx],
                'intent': self.intent_names[self.train_labels[idx]],
                'similarity': float(sim_score)
            })

        # Applica la correzione basata sulla similarità
        corrected_intent, was_corrected = self.get_corrected_intent(
            similar_examples,
            initial_intent,
            similarity_threshold,
            consensus_threshold
        )

        result = {
            'predicted_intent': corrected_intent,
            'original_intent': initial_intent if was_corrected else None,
            'confidence': float(confidence),
            'was_corrected': was_corrected,
            'similar_examples': similar_examples
        }

        return result

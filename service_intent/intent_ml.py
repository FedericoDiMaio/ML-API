from typing import Dict, List, Any

import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from logger import logger


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

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Predizione con regressione logistica
        example_vectorized = self.vectorizer.transform([text])
        predicted_label = self.model.predict(example_vectorized)[0]
        confidence = self.model.predict_proba(example_vectorized).max()
        predicted_intent = self.intent_names[predicted_label]

        result = {
            'predicted_intent': predicted_intent,
            'confidence': float(confidence)
        }

        logger.info(f"Predicted intent: {predicted_intent}")
        logger.info(f"Confidence: {confidence}")

        return result
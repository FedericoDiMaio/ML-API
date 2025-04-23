from typing import Dict, List, Any
import openai
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from config import settings
from logger import logger

vectorize = openai.AzureOpenAI(
    azure_deployment=settings.embeddings_openai_deployment,
    azure_endpoint=settings.embeddings_openai_endpoint,
    api_key=settings.embeddings_openai_key,
    api_version=settings.openai_api_version,
)


def get_embedding(text: str) -> List[float]:
    try:
        response = vectorize.embeddings.create(
            input=text,
            model=settings.embeddings_openai_deployment
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Errore nell'ottenere embedding: {str(e)}")
        return [0.0] * 1536


class IntentClassifier:
    def __init__(self, train_dir: str, batch_size: int = 16, seed: int = 42):
        self.batch_size = batch_size
        self.seed = seed
        self.train_dir = train_dir
        self.train_texts: List[str] = []
        self.train_labels: List[int] = []
        self.intent_names: List[str] = []
        self.intent_embeddings: Dict[str, List[List[float]]] = {}  # Conserva gli embedding per ogni intent
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

        for intent in self.intent_names:
            self.intent_embeddings[intent] = []

        for text_batch, label_batch in raw_train_ds:
            decoded_texts = [text.numpy().decode('utf-8') for text in text_batch]
            labels = label_batch.numpy()

            for text, label in zip(decoded_texts, labels):
                self.train_texts.append(text)
                self.train_labels.append(label)
                intent_name = self.intent_names[label]
                # Salva il testo per questo intent (servirà per il train)
                self.intent_embeddings[intent_name].append(text)

    def train(self) -> None:
        if not self.train_texts:
            raise ValueError("No training data loaded. Call load_data() first.")

        logger.info("Generazione degli embedding per ogni intent...")

        for intent, texts in self.intent_embeddings.items():
            logger.info(f"Generazione embedding per l'intent '{intent}' ({len(texts)} esempi)")
            embeddings = [get_embedding(text) for text in texts]
            self.intent_embeddings[intent] = embeddings

        self.is_trained = True
        logger.info("Generazione degli embedding completata!")

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        input_embedding = get_embedding(text)

        # Calcola la similarità con tutti gli embedding di training
        max_similarity = -1
        predicted_intent = None

        # Dictionary per tenere traccia della similarità media per intent
        intent_similarities = {intent: [] for intent in self.intent_names}

        # Per ogni intent, calcola la similarità con tutti gli esempi
        for intent, embeddings in self.intent_embeddings.items():
            for emb in embeddings:
                # Calcola la similarità coseno
                similarity = cosine_similarity([input_embedding], [emb])[0][0]
                intent_similarities[intent].append(similarity)

        # Calcola la similarità media per ogni intent
        avg_similarities = {}
        for intent, similarities in intent_similarities.items():
            if similarities:  # Se ci sono valori
                avg_similarities[intent] = sum(similarities) / len(similarities)
            else:
                avg_similarities[intent] = 0

        # Trova l'intent con la massima similarità media
        predicted_intent = max(avg_similarities, key=avg_similarities.get)
        confidence = avg_similarities[predicted_intent]

        # Loggare i risultati per il debug
        logger.info(f"Input text: '{text}'")
        for intent, similarity in avg_similarities.items():
            logger.info(f"Similarità media con '{intent}': {similarity:.4f}")

        result = {
            'predicted_intent': predicted_intent,
            'confidence': float(confidence)
        }

        logger.info(f"Predicted intent: {predicted_intent}")
        logger.info(f"Confidence: {confidence}")

        return result
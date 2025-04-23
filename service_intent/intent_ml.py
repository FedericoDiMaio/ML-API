import json
from typing import Dict, List, Any
import openai
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


class IntentClassifier:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.flussi: List[Dict[str, Any]] = []
        self.domande_embeddings: List[Dict[str, Any]] = []  # Lista di tutte le domande con i loro embedding
        self.is_trained = False

    def load_data(self) -> None:
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.flussi = data.get('flussi', [])

            if not self.flussi:
                raise ValueError("Il file JSON non contiene flussi o ha un formato non valido")

            # Conta gli intent unici per logging
            intent_set = set()
            for flusso in self.flussi:
                intent = flusso.get('Intent')
                if intent:
                    intent_set.add(intent)

            logger.info(f"Caricati {len(self.flussi)} flussi con {len(intent_set)} intent diversi")

        except Exception as e:
            logger.error(f"Errore nel caricamento del file JSON: {str(e)}")
            raise

    def train(self) -> None:
        if not self.flussi:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Generazione degli embedding per ogni domanda...")

        for flusso in self.flussi:
            intent = flusso.get('Intent')
            domanda = flusso.get('Domanda')
            risposta = flusso.get('Risposta')
            variabili = flusso.get('Variabili Coinvolte', [])

            if not all([intent, domanda, risposta]):
                logger.warning(f"Flusso incompleto trovato, saltato: {flusso}")
                continue

            # Genera l'embedding per la domanda
            embedding = get_embedding(domanda)

            # Salva i dati completi con l'embedding
            self.domande_embeddings.append({
                'intent': intent,
                'domanda': domanda,
                'risposta': risposta,
                'variabili': variabili,
                'embedding': embedding
            })

        self.is_trained = True
        logger.info(f"Generazione degli embedding completata per {len(self.domande_embeddings)} domande!")

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if not self.domande_embeddings:
            raise ValueError("No training data available.")

        # Genera l'embedding per il testo di input
        input_embedding = get_embedding(text)

        # Calcola la similarità con tutti gli embedding delle domande
        max_similarity = -1
        best_match = None

        for item in self.domande_embeddings:
            # Calcola la similarità coseno
            similarity = cosine_similarity([input_embedding], [item['embedding']])[0][0]

            # Tieni traccia della migliore corrispondenza
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = item

        if not best_match:
            logger.warning("Nessuna corrispondenza trovata.")
            return {
                'predicted_intent': None,
                'confidence': 0.0,
                'risposta': None,
                'domanda_simile': None,
                'variabili_coinvolte': []
            }

        # Loggare i risultati per il debug
        logger.info(f"Input text: '{text}'")
        logger.info(f"Domanda più simile: '{best_match['domanda']}'")
        logger.info(f"Intent associato: '{best_match['intent']}'")
        logger.info(f"Similarità: {max_similarity:.4f}")

        # Crea e restituisci il risultato (aggiunto variabili_coinvolte)
        result = {
            'predicted_intent': best_match['intent'],
            'confidence': float(max_similarity),
            'risposta': best_match['risposta'],
            'domanda_simile': best_match['domanda'],
            'variabili_coinvolte': best_match['variabili']
        }

        return result  # Restituisci il risultato
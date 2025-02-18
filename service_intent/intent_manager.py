from logger import logger
from .intent_ml import IntentClassifier
from .singleton import Singleton


def get_classifier(train_dir: str) -> IntentClassifier:
    classifier = Singleton.get_instance(IntentClassifier, train_dir)

    if not classifier.is_trained:
        logger.info("first API call: Loading and training the model...")
        classifier.load_data()
        classifier.train()
        logger.info("Model training completed!")

    return classifier


def reset_classifier() -> None:
    Singleton.clear_instance(IntentClassifier)

import os

import tensorflow as tf
# from matplotlib import plot_learning_curves
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "data")
train_dir = os.path.join(dataset_dir, "train_intent")
test_dir = os.path.join(dataset_dir, "test")

batch_size = 16
seed = 42

#caricamento dati
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    directory=train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    seed=seed,
    subset='training',
    shuffle=True
)
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    directory=train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    seed=seed,
    subset='validation',
    shuffle=True
)

train_texts = []
train_labels = []

for text_batch, label_batch in raw_train_ds:
    decoded_texts = [text.numpy().decode('utf-8') for text in text_batch]  #decodifica dei testi
    train_texts.extend(decoded_texts)
    train_labels.extend(label_batch.numpy())

val_texts = []
val_labels = []

for text_batch, label_batch in raw_val_ds:
    decoded_texts = [text.numpy().decode('utf-8') for text in text_batch]
    #logger.info(f"Decoded texts: {decoded_texts}")
    val_texts.extend(decoded_texts)
    #logger.info(f"Val texts: {val_texts}")
    val_labels.extend(label_batch.numpy())
    #logger.info(f"Val labels: {val_labels}")

vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

model = LogisticRegression(max_iter=500)
model.fit(X_train, train_labels)
predictions = model.predict(X_val)
# print("\nValutazione sul validation set:")
# print(classification_report(val_labels, predictions))

#plot_learning_curves(model, X_train, train_labels)
intent_names = raw_train_ds.class_names

text = ["che previsioni danno domani a roma?"]

example_vectorized = vectorizer.transform(text)
predicted_label = model.predict(example_vectorized)[0]
predicted_intent = intent_names[predicted_label]
confidence = model.predict_proba(example_vectorized).max()

print(f"\nIntent previsto: {predicted_intent}")
print(f"Confidenza: {confidence:.2f}")

intent_to_api = {
    "get_weather": "/api/weather",
    "get_time": "/api/time",
    "get_news": "/api/news",
}

api_endpoint = intent_to_api.get(predicted_intent, "unknown")
print(f"API da chiamare: {api_endpoint}")

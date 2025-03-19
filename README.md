📊 Machine Learning API - Supervised and Unsupervised Algorithms

This repository offers a comprehensive collection of APIs for Machine Learning algorithms, covering both supervised and unsupervised learning. Ideal for developers, data scientists, and researchers, it provides flexible and high-performance tools to build predictive models and analyze data efficiently.

📝 Intent Classification API @app.post("/predict")
🚀 Description:

This API provides intent classification using Logistic Regression with text preprocessing via TF-IDF (Term Frequency-Inverse Document Frequency). The classifier is trained on labeled text data and predicts the intent of a given input text with a confidence score.
📌 Technique Used:

    Text Vectorization: TF-IDF with bi-grams (ngram_range=(1, 2)) and a maximum of 100 features.
    Classification Model: Logistic Regression with a maximum of 500 iterations (max_iter=500).
    Training Data Handling: Uses tf.keras.utils.text_dataset_from_directory for efficient batch loading and shuffling.
    Training-Validation Split: 80-20 using a fixed seed (seed=42).

📥 Input:

    Raw text string that needs to be classified.

📤 Output:

    Predicted intent label.
    Confidence score of the prediction.

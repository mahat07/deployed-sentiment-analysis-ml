from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string

app = FastAPI()  #initialising fastAPI

#trained model and vectorizer
log_reg_model = joblib.load("logistic_regression_model.joblib")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")

print("Model and Vectorizer Loaded Successfully")

# Request Body Schema
class TextRequest(BaseModel):
    text: str

# Text Preprocessing Function
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\d+", "", text)      # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text

# Prediction Endpoint
@app.post("/predict")
async def predict_sentiment(request: TextRequest):
    text_input = request.text

    # Preprocess
    processed_text = preprocess_text(text_input)

    # Vectorize
    text_tfidf = tfidf_vectorizer.transform([processed_text])

    # Predict
    prediction = log_reg_model.predict(text_tfidf)[0]
    probability = log_reg_model.predict_proba(text_tfidf)[0]

    sentiment_label = "positive" if prediction == 1 else "negative/neutral"

    return {
        "text": text_input,
        "processed_text": processed_text,
        "predicted_sentiment": sentiment_label,
        "prediction_raw_label": int(prediction),
        "probabilities": {
            "negative/neutral": float(probability[0]),
            "positive": float(probability[1])
        }
    }

print("FastAPI app initialized with /predict endpoint.")

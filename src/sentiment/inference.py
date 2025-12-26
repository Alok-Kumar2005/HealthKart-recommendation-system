import os
import sys
import pickle
from src.exception import CustomException
from src.logging import logging


class SentimentInference:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.load_models()

    def load_models(self):
        try:
            with open(os.path.join(self.model_dir, 'svm_model.pkl'), 'rb') as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.model_dir, 'tfidf.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            logging.info("Sentiment models loaded successfully")
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self, text: str):
        try:
            text_tfidf = self.vectorizer.transform([text])
            prediction = self.model.predict(text_tfidf)[0]
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = sentiment_map.get(prediction, "unknown")
            
            try:
                decision_scores = self.model.decision_function(text_tfidf)[0]
                confidence = float(max(abs(decision_scores)))
            except:
                confidence = None
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "text": text
            }
        except Exception as e:
            logging.error(f"Error during sentiment prediction: {e}")
            raise CustomException(e, sys)


if __name__ == '__main__':
    model = SentimentInference()
    result = model.predict("This product is amazing! I love it!")
    print(result)
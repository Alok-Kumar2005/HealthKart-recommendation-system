import os
import sys
import pickle
from src.recommender.recommendation import RecommendationSystem
from src.exception import CustomException
from src.logging import logging


class RecommenderInference:
    def __init__(self, model_dir: str = "models/recommender"):
        self.model_dir = model_dir
        self.load_models()

    def load_models(self):
        try:
            with open(os.path.join(self.model_dir, 'product_features.pkl'), 'rb') as f:
                self.product_features = pickle.load(f)
            with open(os.path.join(self.model_dir, 'similarity_matrix.pkl'), 'rb') as f:
                self.similarity_matrix = pickle.load(f)
            with open(os.path.join(self.model_dir, 'sentiment_scores.pkl'), 'rb') as f:
                self.sentiment_scores = pickle.load(f)
            with open(os.path.join(self.model_dir, 'co_occurrence.pkl'), 'rb') as f:
                self.co_occurrence = pickle.load(f)
            
            # Load temporal analysis data
            try:
                with open(os.path.join(self.model_dir, 'trend_scores.pkl'), 'rb') as f:
                    self.trend_scores = pickle.load(f)
                with open(os.path.join(self.model_dir, 'trend_directions.pkl'), 'rb') as f:
                    self.trend_directions = pickle.load(f)
                logging.info("Temporal analysis data loaded successfully")
            except FileNotFoundError:
                logging.warning("Temporal analysis files not found. Using default values.")
                self.trend_scores = None
                self.trend_directions = None
            
            logging.info("Models loaded successfully")
        except Exception as e:
            raise CustomException(e, sys)
        
    def recommend(self, product_name: str, n_recommendations: int = 10):
        recommender = RecommendationSystem()
        return recommender.getRecommendations(
            product_name,
            self.product_features,
            self.similarity_matrix,
            self.sentiment_scores,
            self.co_occurrence,
            self.trend_scores,
            self.trend_directions,
            n_recommendations=n_recommendations
        )


if __name__ == '__main__':
    model = RecommenderInference()
    res = model.recommend("Pink Friday: Roman Reloaded Re-Up (w/dvd)")
    print(res)
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
        self.trend_scores = None

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
            ## new trend model
            with open(os.path.join(self.model_dir, 'trend_scores.pkl'), 'rb') as f:
                self.trend_scores = pickle.load(f)
                
            
            logging.info("Models loaded.....")
        except Exception as e:
            raise CustomException(e, sys)
        
    def recommend(self, product_name: str, n_recommendations:int = 10):
        recommender = RecommendationSystem()
        recommender.trend_scores = self.trend_scores

        return recommender.getRecommendations(
            product_name,
            self.product_features,
            self.similarity_matrix,
            self.sentiment_scores,
            self.co_occurrence,
            n_recommendations= n_recommendations
        )


if __name__ == '__main__':
    model = RecommenderInference()
    res = model.recommend("Independence Day Resurgence (4k/uhd + Blu-Ray + Digital)")
    print(res)
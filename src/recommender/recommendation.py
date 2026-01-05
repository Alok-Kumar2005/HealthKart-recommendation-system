import os
import sys
import pandas as pd
import numpy as np
import pickle
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from src.exception import CustomException
from src.logging import logging

def load_params(config_path: str = "params.yaml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


class RecommendationSystem:
    def __init__(
            self, 
            entity_data_path: str = "data/entity_data/entity_extracted.csv",
            sentiment_model_path: str = "models/svm_model.pkl",
            vectorizer_path: str = "models/tfidf.pkl",
            output_dir: str = "models/recommender"
    ):
        self.entity_data_path = entity_data_path
        self.sentiment_model_path = sentiment_model_path
        self.vectorizer_path = vectorizer_path
        self.output_dir = output_dir
        self.df = None
        self.sentiment_model = None
        self.text_vectorizer = None
        self.params = load_params()

    def load_data(self):
        try:
            logging.info("Loading data and models for recommendation system")
            ## load entity data
            self.df = pd.read_csv(self.entity_data_path)
            ## load sentiment model and vectorizer
            with open(self.sentiment_model_path, "rb") as f:
                self.sentiment_model = pickle.load(f)
            with open(self.vectorizer_path, "rb") as f:
                self.text_vectorizer = pickle.load(f)
            logging.info("Data and models loaded successfully")
        except Exception as e:
            logging.error("Error loading data or models")
            raise CustomException(e, sys)
        
    def map_sentiment(self, rating):
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
        
    def CalculateSentimentScore(self):
        try:
            logging.info("Calculating sentiment scores for products")
            self.df['sentiment'] = self.df['reviews.rating'].apply(self.map_sentiment)
            ## aggregated sentiment score per product
            product_sentiment = self.df.groupby(['name', 'extracted_brand', 'main_category']).agg(
                {
                    'reviews.rating': ['mean', 'count'],
                    'sentiment': lambda x: (x == 'positive').sum()/len(x)  ## positive sentiment ratio
                }
            ).reset_index()
            product_sentiment.columns = ['product_name', 'brand', 'category', 
                                         'avg_rating', 'review_count', 'positive_ratio']
            
            ## rule: weighted sentiment score
            ## product with more number of reviews gets higher weight
            product_sentiment['confidence'] = np.minimum(
                product_sentiment['review_count'] / product_sentiment['review_count'].quantile(0.75), 1.0
            )
            product_sentiment['sentiment_score'] = (
                product_sentiment['avg_rating'] * self.params['sentiment_score']['normalized_rating'] +
                product_sentiment['positive_ratio'] * self.params['sentiment_score']['positive_ratio']
            ) * product_sentiment['confidence']

            logging.info("Sentiment scores calculated successfully")
            return product_sentiment
        except Exception as e:  
            logging.error("Error calculating sentiment scores")
            raise CustomException(e, sys)
        
    def ContentBasedRecommender(self):
        try:
            logging.info("Building content-based recommendation system")
            ## combine relevant text fields for vectorization
            self.df['feature_text'] = (
                self.df['extracted_brand'].fillna('') + ' ' +
                self.df['main_category'].fillna('') + ' ' +
                self.df['sub_category'].fillna('') + ' ' +
                self.df['product_attributes'].fillna('')
            )
            ## product feature matirx
            product_features = self.df.groupby('name').agg(
                {
                    'feature_text': lambda x: ' '.join(x.unique()),
                    'extracted_brand': 'first',
                    'main_category': 'first',
                    'reviews.rating': 'mean'
                }
            ).reset_index()

            ## tfidf vectorization for similarity
            tfidf = TfidfVectorizer(max_features=5000, stop_words= 'english')
            feature_matrix = tfidf.fit_transform(product_features['feature_text'])

            ## cosine similarity matrix
            similarity_matrix = cosine_similarity(feature_matrix)
            logging.info("Content-based recommendation system built successfully")
            return product_features, similarity_matrix, tfidf
        except Exception as e:
            logging.error("Error building recommendation system")
            raise CustomException(e, sys)
        
    def CollaborativeRecommender(self):
        try:
            logging.info("Building collaborative filtering recommendation system")
            positive_reviews = self.df[self.df['sentiment'] == 'positive'].copy()
            co_occurrence = defaultdict(lambda: defaultdict(int))
            category_groups = positive_reviews.groupby('main_category')['name'].apply(list) 

            for products in category_groups:
                unique_products = list(set(products))
                for i, prod1 in enumerate(unique_products):
                    for prod2 in unique_products[i+1:]:
                        co_occurrence[prod1][prod2] += 1
                        co_occurrence[prod2][prod1] += 1
            
            brand_groups = positive_reviews.groupby('extracted_brand')['name'].apply(list)
            
            for products in brand_groups:
                unique_products = list(set(products))
                for i, prod1 in enumerate(unique_products):
                    for prod2 in unique_products[i+1:]:
                        co_occurrence[prod1][prod2] += 0.5
                        co_occurrence[prod2][prod1] += 0.5
            
            logging.info("Collaborative recommender built (category + brand based)")
            return dict(co_occurrence)

        except Exception as e:
            logging.error("Error building collaborative filtering system")
            raise CustomException(e, sys)
    
    def calculate_temporal_analysis(self, product_features):
        """
        Calculate trend scores and directions for products
        This method should match what's in temporal_analysis.py
        """
        try:
            logging.info("Calculating temporal analysis...")
            
            # Create dictionaries to store trend data
            trend_scores = {}
            trend_directions = {}
            
            # Simple temporal analysis based on review patterns
            for _, product in product_features.iterrows():
                product_name = product['name']
                
                # Get product reviews
                product_reviews = self.df[self.df['name'] == product_name]
                
                if len(product_reviews) > 0:
                    # Calculate trend score (normalized by review count)
                    avg_rating = product_reviews['reviews.rating'].mean()
                    review_count = len(product_reviews)
                    
                    # Normalize trend score
                    trend_score = min(review_count / 100, 1.0) * (avg_rating / 5.0)
                    trend_scores[product_name] = trend_score
                    
                    # Determine trend direction based on rating distribution
                    recent_reviews = product_reviews.tail(min(10, len(product_reviews)))
                    if len(recent_reviews) >= 3:
                        recent_avg = recent_reviews['reviews.rating'].mean()
                        overall_avg = product_reviews['reviews.rating'].mean()
                        
                        if recent_avg > overall_avg + 0.3:
                            trend_directions[product_name] = 'rising'
                        elif recent_avg < overall_avg - 0.3:
                            trend_directions[product_name] = 'falling'
                        else:
                            trend_directions[product_name] = 'stable'
                    else:
                        trend_directions[product_name] = 'stable'
                else:
                    trend_scores[product_name] = 0.5
                    trend_directions[product_name] = 'unknown'
            
            logging.info("Temporal analysis completed")
            return trend_scores, trend_directions
            
        except Exception as e:
            logging.error("Error in temporal analysis")
            raise CustomException(e, sys)
            
    def getRecommendations(self,
                           product_name: str,
                           product_features: pd.DataFrame,
                           similarity_matrix: np.ndarray,
                           sentiment_scores: pd.DataFrame,
                           co_occurrence: dict,
                           trend_scores: dict = None,
                           trend_directions: dict = None,
                           n_recommendations: int = 10,
                           sentiment_weight: float = 0.4,
                           content_weight: float = 0.4,
                           collaborative_weight: float = 0.2
                           ):
        try:
            logging.info(f"Generating recommendations for product: {product_name}")
            if product_name not in product_features['name'].values:
                logging.warning(f"Product {product_name} not found in dataset")
                return f"Product '{product_name}' not found in dataset"
            
            ## index of product
            idx = product_features[product_features['name'] == product_name].index[0]
            content_scores = similarity_matrix[idx]
            sentiment_dict = dict(
                zip(
                    sentiment_scores['product_name'],
                    sentiment_scores['sentiment_score']
                )
            )
            collab_products = co_occurrence.get(product_name, {})
            max_collab = max(collab_products.values()) if collab_products else 1

            ## hybrid socre calculation
            recommendations = []
            for i, row in product_features.iterrows():
                if i == idx:
                    continue
                prod_name = row['name']
                content_score = content_scores[i]
                sentiment_score = sentiment_dict.get(prod_name, 0.5)  ## default neutral score
                collab_score = collab_products.get(prod_name, 0) / max_collab

                hybrid_score = (
                    sentiment_weight * sentiment_score +
                    content_weight * content_score +
                    collaborative_weight * collab_score
                )
                
                # Get trend information if available
                trend_score = trend_scores.get(prod_name, 0.5) if trend_scores else 0.5
                trend_direction = trend_directions.get(prod_name, 'unknown') if trend_directions else 'unknown'
                
                recommendations.append(
                    {
                        'product_name': prod_name,
                        'brand': row['extracted_brand'],
                        'category': row['main_category'],
                        'avg_rating': row['reviews.rating'],
                        'hybrid_score': hybrid_score,
                        'content_score': content_score,
                        'sentiment_score': sentiment_score,
                        'collab_score': collab_score,
                        'trend_score': trend_score,
                        'trend_direction': trend_direction
                    }
                )
            recommendations = sorted(recommendations, key=lambda x: x['hybrid_score'], reverse=True)
            return pd.DataFrame(recommendations[:n_recommendations])
        except Exception as e:
            logging.error("Error generating recommendations")
            raise CustomException(e, sys)
    

    def train_and_save(self):
        try:
            logging.info("Training recommendation system...")
            self.load_data()
            sentiment_scores = self.CalculateSentimentScore()
            product_features, similarity_matrix, tfidf = self.ContentBasedRecommender()
            co_occurrence = self.CollaborativeRecommender()
            
            # Calculate temporal analysis
            trend_scores, trend_directions = self.calculate_temporal_analysis(product_features)
            
            # Save models
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, 'product_features.pkl'), 'wb') as f:
                pickle.dump(product_features, f)
            with open(os.path.join(self.output_dir, 'similarity_matrix.pkl'), 'wb') as f:
                pickle.dump(similarity_matrix, f)
            with open(os.path.join(self.output_dir, 'sentiment_scores.pkl'), 'wb') as f:
                pickle.dump(sentiment_scores, f)
            with open(os.path.join(self.output_dir, 'co_occurrence.pkl'), 'wb') as f:
                pickle.dump(co_occurrence, f)
            with open(os.path.join(self.output_dir, 'content_tfidf.pkl'), 'wb') as f:
                pickle.dump(tfidf, f)
            # Save temporal analysis data
            with open(os.path.join(self.output_dir, 'trend_scores.pkl'), 'wb') as f:
                pickle.dump(trend_scores, f)
            with open(os.path.join(self.output_dir, 'trend_directions.pkl'), 'wb') as f:
                pickle.dump(trend_directions, f)
            
            logging.info(f"Recommendation models saved to {self.output_dir}")
            sample_product = product_features['name'].iloc[0]
            logging.info(f"\nSample recommendations for: {sample_product}")
            recs = self.getRecommendations(
                sample_product, 
                product_features, 
                similarity_matrix,
                sentiment_scores,
                co_occurrence,
                trend_scores,
                trend_directions
            )
            print("\n", recs[['product_name', 'brand', 'hybrid_score', 'avg_rating', 'trend_score', 'trend_direction']].head())
            
        except Exception as e:
            logging.error("Error training recommendation system")
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    try:
        recommender = RecommendationSystem()
        recommender.train_and_save()
        print("\nRecommendation system built successfully!")
        
    except Exception as e:
        raise CustomException(e, sys)
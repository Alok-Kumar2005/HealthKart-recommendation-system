import os
import sys
import pandas as pd
import numpy as np
import pickle
import yaml
from datetime import datetime, timedelta
from src.exception import CustomException
from src.logging import logging


class TemporalSentimentAnalysis:
    def __init__(self, entity_data_path: str = "data/entity_data/entity_extracted2.csv", output_dir: str = "models/recommender"):
        self.entity_data_path = entity_data_path
        self.output_dir = output_dir
        self.df = None
        self.params = self.load_params()

    def load_params(self):
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f)

    def load_data(self):
        try:
            logging.info("Loading entity extracted data...")
            self.df = pd.read_csv(self.entity_data_path)
            self.df['review_date'] = pd.to_datetime(self.df['reviews.date'], errors='coerce', utc=True)

            ## Drop rows with invalid dates
            self.df = self.df.dropna(subset=['review_date'])
            logging.info("Data loaded successfully.")
        except Exception as e:
            raise CustomException(f"Error loading data: {e}", sys)
        
    def map_sentiment(self, rating):
        if rating >= 4:
            return 'positive'
        elif rating == 3:
            return 'neutral'
        else:
            return 'negative'
        
    def calculate_sentiment_trends(self):
        try:
            logging.info("Calculating sentiment trends of each product")
            self.df['sentiment'] = self.df['reviews.rating'].apply(self.map_sentiment)
            ## month year column for grouping
            self.df['review_month'] = self.df['review_date'].dt.to_period('M')

            monthly_trends  = self.df.groupby(['name', 'review_month']).agg({
                'reviews.rating': ['mean', 'count'],
                'sentiment': lambda x: (x == 'positive').sum() / len(x) if len(x) > 0 else 0
            }).reset_index()

            monthly_trends.columns = ['product_name', 'month', 'avg_rating', 'review_count', 'positive_ratio']
            return monthly_trends
        except Exception as e:  
            raise CustomException(f"Error calculating sentiment trends: {e}", sys)
    
    def calculate_trend_score(self, monthly_trends):
        try:
            logging.info("Calculating trend scores for products...")
            product_trends = {}
            
            for product_name, group in monthly_trends.groupby('product_name'):
                if len(group)< 2:
                    product_trends[product_name] = {
                        'trend_direction': 'stable',
                        'trend_score': 0.0,
                        'recent_momentum': 0.0,
                        'months_tracked': len(group)
                    }
                    continue

                group = group.sort_values('month')
                ## linear trend using numpy polyfit
                x = np.arange(len(group))
                y_rating = group['avg_rating'].values
                y_positive = group['positive_ratio'].values

                ## fut linear trend
                rating_slope = np.polyfit(x, y_rating, 1)[0] if len(x) > 1 else 0
                positive_slope = np.polyfit(x, y_positive, 1)[0] if len(x) > 1 else 0

                ## combine slopes for overall trend
                trend_score = (rating_slope * self.params['temporal_sentiment']['rating_slope'] + positive_slope * self.params['temporal_sentiment']['positive_slope'])

                if len(group) >= 6:
                    recent_avg = group.tail(3)['avg_rating'].mean()
                    previous_avg = group.head(len(group)-3)['avg_rating'].mean()
                    recent_momentum = recent_avg - previous_avg
                else:
                    recent_momentum = trend_score
                
                # Determine direction
                if trend_score > 0.05:
                    direction = 'improving'
                elif trend_score < -0.05:
                    direction = 'declining'
                else:
                    direction = 'stable'
                
                product_trends[product_name] = {
                    'trend_direction': direction,
                    'trend_score': float(trend_score),
                    'recent_momentum': float(recent_momentum),
                    'rating_slope': float(rating_slope),
                    'positive_slope': float(positive_slope),
                    'months_tracked': len(group),
                    'latest_rating': float(group.iloc[-1]['avg_rating']),
                    'earliest_rating': float(group.iloc[0]['avg_rating'])
                }
            # Distribution stats
            improving = sum(1 for v in product_trends.values() if v['trend_direction'] == 'improving')
            declining = sum(1 for v in product_trends.values() if v['trend_direction'] == 'declining')
            stable = sum(1 for v in product_trends.values() if v['trend_direction'] == 'stable')
            
            logging.info(f"Trend distribution - Improving: {improving}, Declining: {declining}, Stable: {stable}")
            
            return product_trends
        except Exception as e:
            raise CustomException(f"Error calculating trend scores: {e}", sys)
        
    def calculate_recency_weights(self):
        try:
            logging.info("Calculating recency weights")
            max_date = self.df['review_date'].max()
            ## days since review
            self.df['days_since_review'] = (max_date - self.df['review_date']).dt.days

            ## time decay: recent reviews have higher weight
            decay_constant = self.params['temporal_sentiment']['decay_constant']
            self.df['recency_weight'] = np.exp(-self.df['days_since_review'] / decay_constant)

            return self.df
        except Exception as e:
            raise CustomException(f"Error calculating recency weights: {e}", sys)
        
    def calculate_weighted_sentiment_scores(self):
        try:
            logging.info("Calculating weighted sentiment scores...")
            self.df = self.calculate_recency_weights()
            weighted_scores = self.df.groupby(['name', 'extracted_brand', 'main_category']).apply(
                lambda x: pd.Series({
                    'weighted_avg_rating': (x['reviews.rating'] * x['recency_weight']).sum() / x['recency_weight'].sum(),
                    'weighted_positive_ratio': ((x['sentiment'] == 'positive') * x['recency_weight']).sum() / x['recency_weight'].sum(),
                    'total_reviews': len(x),
                    'recent_reviews': len(x[x['days_since_review'] <= 90]),  # reviews in last 90 days
                    'avg_rating_all_time': x['reviews.rating'].mean(),
                    'avg_rating_recent': x[x['days_since_review'] <= 90]['reviews.rating'].mean() if len(x[x['days_since_review'] <= 90]) > 0 else x['reviews.rating'].mean()
                })
            ).reset_index()
            
            weighted_scores.columns = [
                'product_name', 'brand', 'category', 'weighted_avg_rating',
                'weighted_positive_ratio', 'total_reviews', 'recent_reviews',
                'avg_rating_all_time', 'avg_rating_recent'
            ]
            # Calculate confidence based on review count
            weighted_scores['confidence'] = np.minimum(
                weighted_scores['total_reviews'] / weighted_scores['total_reviews'].quantile(0.75),
                1.0
            )
            
            # Final weighted sentiment score
            weighted_scores['weighted_sentiment_score'] = (
                weighted_scores['weighted_avg_rating'] * 0.6 +
                weighted_scores['weighted_positive_ratio'] * 0.4
            ) * weighted_scores['confidence']
            
            logging.info("Weighted sentiment scores calculated")
            
            return weighted_scores
        except Exception as e:
            raise CustomException(f"Error calculating weighted sentiment scores: {e}", sys)
        
    def analyze_and_save(self):
        try:
            logging.info("Starting temporal sentiment analysis...")
            self.load_data()
            monthly_trends = self.calculate_sentiment_trends()
            trend_scores = self.calculate_trend_score(monthly_trends)
            weighted_sentiment = self.calculate_weighted_sentiment_scores()

            os.makedirs(self.output_dir, exist_ok=True)
            monthly_trends.to_csv(
                os.path.join(self.output_dir, 'monthly_sentiment_trends.csv'),
                index=False
            )
            # Save trend scores
            with open(os.path.join(self.output_dir, 'trend_scores.pkl'), 'wb') as f:
                pickle.dump(trend_scores, f)
            
            # Save weighted sentiment
            with open(os.path.join(self.output_dir, 'weighted_sentiment_scores.pkl'), 'wb') as f:
                pickle.dump(weighted_sentiment, f)
            
            self.generate_summary_report(trend_scores, weighted_sentiment)
            return trend_scores, weighted_sentiment
        except Exception as e:
            raise CustomException(f"Error in analyze_and_save: {e}", sys)

    def generate_summary_report(self, trend_scores, weighted_sentiment):
        """Generate human-readable summary report"""
        try:
            report_path = os.path.join(self.output_dir, 'temporal_analysis_report.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("TEMPORAL SENTIMENT ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                # Overall statistics
                f.write("OVERALL STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Products Analyzed: {len(trend_scores)}\n")
                
                improving = sum(1 for v in trend_scores.values() if v['trend_direction'] == 'improving')
                declining = sum(1 for v in trend_scores.values() if v['trend_direction'] == 'declining')
                stable = sum(1 for v in trend_scores.values() if v['trend_direction'] == 'stable')
                
                f.write(f"Improving Sentiment: {improving} ({improving/len(trend_scores)*100:.1f}%)\n")
                f.write(f"Declining Sentiment: {declining} ({declining/len(trend_scores)*100:.1f}%)\n")
                f.write(f"Stable Sentiment: {stable} ({stable/len(trend_scores)*100:.1f}%)\n\n")
                
                # Top improving products
                f.write("TOP 10 IMPROVING PRODUCTS\n")
                f.write("-" * 80 + "\n")
                sorted_improving = sorted(
                    [(k, v) for k, v in trend_scores.items() if v['trend_direction'] == 'improving'],
                    key=lambda x: x[1]['trend_score'],
                    reverse=True
                )[:10]
                
                for i, (product, data) in enumerate(sorted_improving, 1):
                    f.write(f"{i}. {product}\n")
                    f.write(f"   Trend Score: {data['trend_score']:.4f}\n")
                    f.write(f"   Rating: {data['earliest_rating']:.2f} → {data['latest_rating']:.2f}\n")
                    f.write(f"   Months Tracked: {data['months_tracked']}\n\n")
                
                # Top declining products
                f.write("TOP 10 DECLINING PRODUCTS\n")
                f.write("-" * 80 + "\n")
                sorted_declining = sorted(
                    [(k, v) for k, v in trend_scores.items() if v['trend_direction'] == 'declining'],
                    key=lambda x: x[1]['trend_score']
                )[:10]
                
                for i, (product, data) in enumerate(sorted_declining, 1):
                    f.write(f"{i}. {product}\n")
                    f.write(f"   Trend Score: {data['trend_score']:.4f}\n")
                    f.write(f"   Rating: {data['earliest_rating']:.2f} → {data['latest_rating']:.2f}\n")
                    f.write(f"   Months Tracked: {data['months_tracked']}\n\n")
                
                # Recency impact
                f.write("RECENCY WEIGHTING IMPACT\n")
                f.write("-" * 80 + "\n")
                
                # Products where recent reviews differ significantly from all-time
                weighted_sentiment['rating_delta'] = weighted_sentiment['avg_rating_recent'] - weighted_sentiment['avg_rating_all_time']
                top_positive_delta = weighted_sentiment.nlargest(5, 'rating_delta')
                top_negative_delta = weighted_sentiment.nsmallest(5, 'rating_delta')
                
                f.write("Products with IMPROVING recent reviews:\n")
                for idx, row in top_positive_delta.iterrows():
                    f.write(f"  - {row['product_name']}\n")
                    f.write(f"    All-time: {row['avg_rating_all_time']:.2f}, Recent: {row['avg_rating_recent']:.2f}\n")
                
                f.write("\nProducts with DECLINING recent reviews:\n")
                for idx, row in top_negative_delta.iterrows():
                    f.write(f"  - {row['product_name']}\n")
                    f.write(f"    All-time: {row['avg_rating_all_time']:.2f}, Recent: {row['avg_rating_recent']:.2f}\n")
            
            logging.info(f"Summary report saved to {report_path}")
        
        except Exception as e:
            logging.error("Error generating summary report")
            raise CustomException(e, sys)


        
if __name__ == "__main__":
    analyzer = TemporalSentimentAnalysis()
    trend_scores, weighted_sentiment = analyzer.analyze_and_save()
import os
import sys
import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logging import logging


class FeatureEngineering:
    def __init__(
        self,
        preprocessed_data_path: str = "data/preprocessed_data/preprocessed_reviews.csv",
        output_dir: str = "data/final_data"
    ):
        self.preprocessed_data_path = preprocessed_data_path
        self.output_dir = output_dir

    @staticmethod
    def load_params(config_path: str = "params.yaml") -> dict:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    @staticmethod
    def map_sentiment(rating):
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"

    def engineer_features(self):
        try:
            logging.info("Starting feature engineering")
            df = pd.read_csv(self.preprocessed_data_path)
            logging.info(f"Loaded preprocessed data from {self.preprocessed_data_path}")
            df["Review"] = df["reviews.rating"].apply(self.map_sentiment)
            le = LabelEncoder()
            df["Review_encoded"] = le.fit_transform(df["Review"])
            df = df.drop(columns=["reviews.rating", "Review"])
            logging.info("Feature engineering completed")
            return df
        except Exception as e:
            logging.error("Error occurred during feature engineering")
            raise CustomException(e, sys)

    def split_and_save(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        try:
            logging.info("Starting train-test split")

            X = df.drop(columns=["Review_encoded"])
            y = df["Review_encoded"]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size= test_size,
                random_state= random_state,
                stratify=y,
            )
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
            os.makedirs(self.output_dir, exist_ok=True)

            train_path = os.path.join(self.output_dir, "featured_reviews_train.csv")
            test_path = os.path.join(self.output_dir, "featured_reviews_test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logging.info(f"Training data saved to {train_path}")
            logging.info(f"Test data saved to {test_path}")
            logging.info("Train data size : {}, Test data size : {}".format(train_df.shape, test_df.shape))

        except Exception as e:
            logging.error("Error occurred during train-test split or saving data")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        feature_engineer = FeatureEngineering()
        params = feature_engineer.load_params()
        test_size = params["feature_engineering"]["test_size"]
        random_state = params["feature_engineering"]["random_state"]

        df_featured = feature_engineer.engineer_features()
        feature_engineer.split_and_save(df_featured, test_size=test_size, random_state=random_state)

        print("Feature engineering and data split completed successfully")

    except Exception as e:
        raise CustomException(e, sys)

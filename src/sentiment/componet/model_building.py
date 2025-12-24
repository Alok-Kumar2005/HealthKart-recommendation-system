import os
import sys
import yaml
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from src.exception import CustomException
from src.logging import logging


class ModelTrainer:
    def __init__(
        self,
        train_data_path: str = "data/final_data/featured_reviews_train.csv",
        test_data_path: str = "data/final_data/featured_reviews_test.csv",
        model_dir: str = "models",
        params_path: str = "params.yaml",
    ):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.model_dir = model_dir
        self.params_path = params_path

    def load_params(self):
        try:
            with open(self.params_path, "r") as file:
                params = yaml.safe_load(file)
            logging.info("Model parameters loaded successfully")
            return params
        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self):
        try:
            train_df = pd.read_csv(self.train_data_path).dropna()
            test_df = pd.read_csv(self.test_data_path).dropna()

            X_train = train_df["transformed_text"]
            y_train = train_df["Review_encoded"]

            X_test = test_df["transformed_text"]
            y_test = test_df["Review_encoded"]

            logging.info("Train and test data loaded successfully")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)

    def train(self):
        try:
            logging.info("Starting model training")
            params = self.load_params()
            X_train, X_test, y_train, y_test = self.load_data()
            tfidf = TfidfVectorizer(
                max_features=params["tfidf"]["max_features"],
                ngram_range=tuple(params["tfidf"]["ngram_range"]),
            )

            X_train_tfidf = tfidf.fit_transform(X_train)
            X_test_tfidf = tfidf.transform(X_test)

            svm_model = LinearSVC(
                C=params["model"]["C"],
                max_iter=params["model"]["max_iter"],
                class_weight=params["model"]["class_weight"],
            )

            svm_model.fit(X_train_tfidf, y_train)
            y_pred = svm_model.predict(X_test_tfidf)
            report = classification_report(y_test, y_pred)
            logging.info(f"\n{report}")

            # Save model & vectorizer
            os.makedirs(self.model_dir, exist_ok=True)
            with open(os.path.join(self.model_dir, "tfidf.pkl"), "wb") as f:
                pickle.dump(tfidf, f)

            with open(os.path.join(self.model_dir, "svm_model.pkl"), "wb") as f:
                pickle.dump(svm_model, f)

            logging.info("Model and vectorizer saved successfully")

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        trainer.train()
        print("Model training completed successfully")

    except Exception as e:
        raise CustomException(e, sys)

import sys
import os
import joblib
import yaml
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from src.exception import CustomException
from src.logging import logging


class ModelEvaluation:
    def __init__(self):
        try:
            with open("params.yaml", "r") as f:
                self.params = yaml.safe_load(f)

            self.test_data_path = "data/featured_data/featured_reviews.csv"
            self.model_path = "models/svm_model.pkl"
            self.vectorizer_path = "models/tfidf.pkl"
            self.metrics_path = "models/metrics"

            os.makedirs(self.metrics_path, exist_ok=True)

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate(self):
        try:
            logging.info("Starting model evaluation")
            df_test = pd.read_csv(self.test_data_path)
            df_test = df_test.dropna()
            X_test = df_test["transformed_text"]
            y_test = df_test["Review_encoded"]

            # Load model and vectorizer
            model = joblib.load(self.model_path)
            tfidf = joblib.load(self.vectorizer_path)

            # Transform text
            X_test_tfidf = tfidf.transform(X_test)
            y_pred = model.predict(X_test_tfidf)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics_file = os.path.join(self.metrics_path, "metrics.yaml")
            with open(metrics_file, "w") as f:
                yaml.dump(
                    {
                        "accuracy": float(acc),
                        "classification_report": report
                    },
                    f
                )

            logging.info(f"Model evaluation completed. Accuracy: {acc}")
            logging.info(f"Metrics saved at {metrics_file}")

        except Exception as e:
            logging.error("Error during model evaluation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.evaluate()

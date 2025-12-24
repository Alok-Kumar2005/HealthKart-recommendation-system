import os
import sys
import pandas as pd
import nltk

from src.exception import CustomException
from src.logging import logging

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class DataPreprocessing:
    def __init__(
        self,
        raw_data_path: str = "data/raw/dataset.csv",
        output_dir: str = "data/preprocessed_data",
        output_file: str = "preprocessed_reviews.csv",
    ):
        self.raw_data_path = raw_data_path
        self.output_path = output_dir
        self.output_file = os.path.join(output_dir, output_file)
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    @staticmethod
    def download_nltk_resources():
        nltk.download("punkt")
        nltk.download("stopwords")

    def transform_text(self, text: str) -> str:
        try: 
            logging.info("Transforming text data")
            if not isinstance(text, str):
                return ""

            text = text.lower()
            tokens = nltk.word_tokenize(text)
            tokens = [word for word in tokens if word.isalnum()]
            tokens = [word for word in tokens if word not in self.stop_words]
            tokens = [self.ps.stem(word) for word in tokens]

            return " ".join(tokens)
        except Exception as e:
            logging.error("Error occurred during text transformation")
            raise CustomException(e, sys)

    def preprocess(self) -> pd.DataFrame:
        try:
            logging.info("Starting data preprocessing")
            df = pd.read_csv(self.raw_data_path)
            logging.info(f"Raw data loaded from {self.raw_data_path}")
            df = df[["reviews.text", "reviews.rating"]]
            df = df.dropna()
            df = df.drop_duplicates()

            logging.info("Dropped NA and duplicate rows")
            df["transformed_text"] = df["reviews.text"].apply(self.transform_text)
            df = df.drop(columns=["reviews.text"])
            logging.info("Text preprocessing completed")
            return df
        except Exception as e:
            logging.error("Error occurred during data preprocessing")
            raise CustomException(e, sys)

    def save_preprocessed_data(self, df: pd.DataFrame):
        try:
            os.makedirs(self.output_path, exist_ok=True)
            df.to_csv(self.output_file, index=False)

            logging.info(f"Preprocessed data saved to {self.output_file}")

        except Exception as e:
            logging.error("Error occurred while saving preprocessed data")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        DataPreprocessing.download_nltk_resources()

        processor = DataPreprocessing()
        processed_df = processor.preprocess()
        processor.save_preprocessed_data(processed_df)

        print("Preprocessed data saved successfully")

    except Exception as e:
        raise CustomException(e, sys)

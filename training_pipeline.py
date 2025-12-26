import sys
import os
from src.logging import logging
from src.exception import CustomException
from src.sentiment.componet.data_ingestion import DataIngestion
from src.sentiment.componet.data_preprocessing import DataPreprocessing
from src.sentiment.componet.feature_engineering import FeatureEngineering
from src.sentiment.componet.model_building import ModelTrainer
from src.entity.entity_extraction import EntityExtraction
from src.recommender.recommendation import RecommendationSystem


def run_pipeline():
    try:
        logging.info("="*80)
        logging.info("STARTING HEALTHKART RECOMMENDATION SYSTEM TRAINING PIPELINE")
        logging.info("="*80)
        logging.info("\n" + "="*80)
        logging.info("STEP 1: DATA INGESTION")
        logging.info("="*80)
        
        file_id = "1ShhXrwl89sZmwddLg1H8eHyW0IlmyKhY"
        data_ingestion = DataIngestion(file_id=file_id)
        df = data_ingestion.download_data()
        logging.info(f"✓ Data ingestion completed. Shape: {df.shape}")
        
        logging.info("\n" + "="*80)
        logging.info("STEP 2: DATA PREPROCESSING")
        logging.info("="*80)
        
        DataPreprocessing.download_nltk_resources()
        preprocessor = DataPreprocessing()
        processed_df = preprocessor.preprocess()
        preprocessor.save_preprocessed_data(processed_df)
        logging.info(f"✓ Data preprocessing completed. Shape: {processed_df.shape}")
        
        logging.info("\n" + "="*80)
        logging.info("STEP 3: FEATURE ENGINEERING")
        logging.info("="*80)
        
        feature_engineer = FeatureEngineering()
        params = feature_engineer.load_params()
        df_featured = feature_engineer.engineer_features()
        feature_engineer.split_and_save(
            df_featured,
            test_size=params["feature_engineering"]["test_size"],
            random_state=params["feature_engineering"]["random_state"]
        )
        logging.info("✓ Feature engineering and train-test split completed")
        
        logging.info("\n" + "="*80)
        logging.info("STEP 4: SENTIMENT MODEL TRAINING")
        logging.info("="*80)
        
        trainer = ModelTrainer()
        trainer.train()
        logging.info("✓ Sentiment model training completed")
        
        logging.info("\n" + "="*80)
        logging.info("STEP 5: ENTITY EXTRACTION")
        logging.info("="*80)
        
        extractor = EntityExtraction()
        extracted_df = extractor.process_data()
        extractor.save_data(extracted_df)
        logging.info(f"✓ Entity extraction completed. Shape: {extracted_df.shape}")
        
        logging.info("\n" + "="*80)
        logging.info("STEP 6: RECOMMENDATION SYSTEM TRAINING")
        logging.info("="*80)
        
        recommender = RecommendationSystem()
        recommender.train_and_save()
        logging.info("✓ Recommendation system training completed")
        
        logging.info("\n" + "="*80)
        logging.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("="*80)
        logging.info("\nAll models have been trained and saved.")
        logging.info("\nNext steps:")
        logging.info("1. Start the FastAPI server: uvicorn app:app --reload")
        logging.info("2. Or use Docker: docker-compose up --build")
        logging.info("3. Access API docs at: http://localhost:8000/docs")
        
        return True
        
    except CustomException as e:
        logging.error(f"\nPipeline failed with custom exception: {e}")
        return False
    except Exception as e:
        logging.error(f"\nPipeline failed with exception: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HEALTHKART RECOMMENDATION SYSTEM - TRAINING PIPELINE")
    print("="*80)
    print("\nThis will run the complete training pipeline:")
    print("1. Data Ingestion")
    print("2. Data Preprocessing")
    print("3. Feature Engineering")
    print("4. Sentiment Model Training")
    print("5. Entity Extraction")
    print("6. Recommendation System Training")
    print("\nThis may take several minutes to complete...")
    print("="*80 + "\n")
    
    success = run_pipeline()
    
    if success:
        print("\n" + "="*80)
        print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("TRAINING PIPELINE FAILED!")
        print("="*80)
        sys.exit(1)
import os
import sys
import pandas as pd
import spacy
import re
from collections import Counter
from src.exception import CustomException
from src.logging import logging


class EntityExtraction:
    def __init__(
        self,
        data_path: str = "data/raw/dataset.csv",
        output_dir: str = "data/entity_data",
        output_file: str = "entity_extracted2.csv"
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, output_file)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logging.info("SpaCy model loaded successfully")
        except:
            logging.info("Downloading SpaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract_brands(self, row):
        try:
            if pd.notna(row.get('brand')) and row.get('brand').strip():
                return row['brand'].strip()
            
            if pd.notna(row.get('reviews.text')):
                doc = self.nlp(str(row['reviews.text'])[:500])  ## limit to first 500 chars
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT']:
                        return ent.text
            
            return "Unknown"
        except Exception as e:
            logging.error(f"Error in brand extraction: {e}")
            return "Unknown"

    def parse_categories(self, categories_str):
        try:
            if pd.isna(categories_str) or not categories_str:
                return {
                    'main_category': 'Unknown',
                    'sub_category': 'Unknown',
                    'all_categories': []
                }
            
            categories = [cat.strip() for cat in str(categories_str).split(',')]
            
            return {
                'main_category': categories[0] if len(categories) > 0 else 'Unknown',
                'sub_category': categories[1] if len(categories) > 1 else 'Unknown',
                'all_categories': categories
            }
        except Exception as e:
            logging.error(f"Error parsing categories: {e}")
            return {
                'main_category': 'Unknown',
                'sub_category': 'Unknown',
                'all_categories': []
            }

    def extract_product_attributes(self, text):
        try:
            if pd.isna(text):
                return []
            
            attributes = []
            doc = self.nlp(str(text)[:500])
            
            for ent in doc.ents:
                if ent.label_ in ['PRODUCT', 'ORG', 'GPE']:
                    attributes.append(ent.text)
            
            ### common product attribute patterns
            patterns = [
                r'\b(flavor|flavour|taste|scent|color|colour|size)\s+:\s+(\w+)',
                r'\b(\w+)\s+(flavor|flavour|taste|scent)',
                r'\b(organic|natural|sugar-free|gluten-free|vegan)\b'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, str(text), re.IGNORECASE)
                attributes.extend([match if isinstance(match, str) else ' '.join(match) 
                                 for match in matches])
            
            return list(set(attributes))[:5]  ### return top 5 unique attributes
            
        except Exception as e:
            logging.error(f"Error extracting attributes: {e}")
            return []

    def process_data(self):
        try:
            logging.info("Starting entity extraction")
            df = pd.read_csv(self.data_path)
            logging.info(f"Loaded data with {len(df)} rows")
            logging.info("Extracting brands...")
            df['extracted_brand'] = df.apply(self.extract_brands, axis=1)
            # Parse categories
            logging.info("Parsing categories...")
            category_data = df['categories'].apply(self.parse_categories)
            df['main_category'] = category_data.apply(lambda x: x['main_category'])
            df['sub_category'] = category_data.apply(lambda x: x['sub_category'])
            df['all_categories'] = category_data.apply(lambda x: ','.join(x['all_categories']))
            
            # Extract product attributes from text
            logging.info("Extracting product attributes...")
            df['product_attributes'] = df['reviews.text'].apply(
                lambda x: ','.join(self.extract_product_attributes(x))
            )
            
            # Create a clean dataset with relevant columns
            output_df = df[[
                'id', 'name', 'extracted_brand', 'main_category', 
                'sub_category', 'all_categories', 'product_attributes',
                'reviews.text', 'reviews.rating', 'reviews.title', 'reviews.date'
            ]].copy()
            
            logging.info("Entity extraction completed")
            return output_df
            
        except Exception as e:
            logging.error("Error during entity extraction")
            raise CustomException(e, sys)

    def save_data(self, df: pd.DataFrame):
        """Save extracted entities"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            df.to_csv(self.output_file, index=False)
            logging.info(f"Entity data saved to {self.output_file}")
            
            # Save summary statistics
            summary_file = os.path.join(self.output_dir, "entity_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("Entity Extraction Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Records: {len(df)}\n")
                f.write(f"Unique Brands: {df['extracted_brand'].nunique()}\n")
                f.write(f"Unique Main Categories: {df['main_category'].nunique()}\n\n")
                f.write("Top 10 Brands:\n")
                f.write(df['extracted_brand'].value_counts().head(10).to_string())
                f.write("\n\nTop 10 Categories:\n")
                f.write(df['main_category'].value_counts().head(10).to_string())
            
            logging.info(f"Summary saved to {summary_file}")
            
        except Exception as e:
            logging.error("Error saving entity data")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        extractor = EntityExtraction()
        extracted_df = extractor.process_data()
        extractor.save_data(extracted_df)
        print("Entity extraction completed successfully!")
        
    except Exception as e:
        raise CustomException(e, sys)
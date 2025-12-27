# HealthKart Recommendation System

A hybrid recommendation system combining sentiment analysis, content-based filtering, and collaborative filtering to provide personalized product recommendations based on customer reviews.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training Pipeline](#training-pipeline)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)
- [Usage Examples](#usage-examples)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)

---

## Overview

This project implements an end-to-end machine learning system for product recommendations that:
- Analyzes customer sentiment from reviews using NLP
- Extracts entities (brands, categories, attributes) from unstructured text
- Combines multiple recommendation strategies (content-based, collaborative, sentiment-based)
- Provides a production-ready REST API using FastAPI
- Fully containerized with Docker for easy deployment

**Dataset**: [Amazon Customer Reviews Dataset](https://lnkd.in/guXPbKgc)

---

## ✨ Features

### Sentiment Analysis
- Multi-class sentiment classification (Positive, Neutral, Negative)
- TF-IDF vectorization with LinearSVC model
- Confidence scores for predictions
- Text preprocessing with NLTK (tokenization, stemming, stopword removal)

### Entity Extraction
- Brand identification using SpaCy NER
- Category hierarchy parsing (main/sub categories)
- Product attribute extraction from review text
- Automated entity consolidation

### Recommendation System
- **Hybrid Approach**: Combines three recommendation strategies
  - Content-Based (40%): TF-IDF similarity on product features
  - Collaborative Filtering (20%): Co-occurrence patterns from positive reviews
  - Sentiment Scoring (40%): Aggregated review sentiment with confidence weighting
- Configurable weights for each strategy
- Returns ranked recommendations with detailed scores

### API Features
- RESTful API with FastAPI
- Interactive API documentation (Swagger UI)
- Input validation with Pydantic
- Comprehensive error handling
- Health check endpoints
- CORS support

---

## Tech Stack

**Machine Learning & NLP**
- scikit-learn (TF-IDF, LinearSVC, Cosine Similarity)
- NLTK (Text preprocessing)
- SpaCy (Named Entity Recognition)
- imbalanced-learn (Handling class imbalance)
- NumPy, Pandas (Data manipulation)

**API & Web Framework**
- FastAPI (REST API)
- Uvicorn (ASGI server)
- Pydantic (Data validation)

**DevOps & Deployment**
- Docker (Containerization)
- Docker Compose (Multi-container orchestration)
- Python 3.10

**Others**
- PyYAML (Configuration management)
- Joblib/Pickle (Model serialization)
- gdown (Google Drive data download)

---

## Project Structure

```
healthkart-recommendation/
│
├── app.py                          # FastAPI application entry point
├── training_pipeline.py            # Complete training pipeline orchestration
├── requirements.txt                # Python dependencies
├── params.yaml                     # Model hyperparameters and configuration
├── Dockerfile                      # Docker image definition
├── docker-compose.yml              # Docker Compose configuration
├── .dockerignore                   # Docker build exclusions
├── .gitignore                      # Git exclusions
│
├── src/
│   ├── __init__.py
│   ├── exception.py                # Custom exception handling
│   ├── logging.py                  # Logging configuration
│   │
│   ├── sentiment/                  # Sentiment Analysis Module
│   │   ├── componet/
│   │   │   ├── data_ingestion.py          # Download dataset from Google Drive
│   │   │   ├── data_preprocessing.py      # Text cleaning and transformation
│   │   │   ├── feature_engineering.py     # Label encoding and train-test split
│   │   │   ├── model_building.py          # TF-IDF + LinearSVC training
│   │   │   └── model_evaluation.py        # Model performance metrics
│   │   │
│   │   └── inference.py            # Sentiment prediction API
│   │
│   ├── entity/
│   │   └── entity_extraction.py    # Brand, category, attribute extraction
│   │
│   └── recommender/
│       ├── recommendation.py       # Hybrid recommendation system training
│       └── inference.py            # Recommendation API
│
├── data/
│   ├── raw/                        # Original dataset
│   ├── preprocessed_data/          # Cleaned and transformed data
│   ├── final_data/                 # Train/test splits
│   └── entity_data/                # Extracted entities
│
├── models/
│   ├── svm_model.pkl               # Trained sentiment classifier
│   ├── tfidf.pkl                   # TF-IDF vectorizer
│   └── recommender/                # Recommendation system artifacts
│       ├── product_features.pkl
│       ├── similarity_matrix.pkl
│       ├── sentiment_scores.pkl
│       ├── co_occurrence.pkl
│       └── content_tfidf.pkl
│
└── logs/                           # Training and API logs
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- 4GB+ RAM recommended
- Git

### Local Setup (Without Docker)

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/healthkart-recommendation.git
cd healthkart-recommendation
```

2. **Create virtual environment**
```bash
uv init
uv venv
.venv\Scripts\activate
```

3. **Install dependencies**
```
uv add -r requirements.txt
```

4. **Download NLTK data and SpaCy model**
```bash
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm
```

5. **Create necessary directories**
```bash
mkdir -p logs models/recommender data/raw data/preprocessed_data data/final_data data/entity_data
```

## Pull image, run and check
```
docker pull alok8090/healthkart-recommendation:v1.0
docker run -p 5000:8000 alok8090/healthkart-recommendation:v1.0
http://localhost:5000/docs
```

---

## 🎓 Training Pipeline

The training pipeline consists of 6 sequential steps that must be run in order:

### Run Complete Pipeline

```bash
python training_pipeline.py
```

This executes all steps automatically:
1. **Data Ingestion**: Downloads dataset from Google Drive
2. **Data Preprocessing**: Cleans text, removes stopwords, applies stemming
3. **Feature Engineering**: Encodes labels, splits train/test data
4. **Sentiment Model Training**: Trains TF-IDF + LinearSVC classifier
5. **Entity Extraction**: Extracts brands, categories, attributes using SpaCy
6. **Recommendation System Training**: Builds hybrid recommendation models

**Training Time**: ~10-15 minutes (depending on hardware)

### Run Individual Components

If you need to run steps separately:

```bash
# Step 1: Data Ingestion
python -c "from src.sentiment.componet.data_ingestion import DataIngestion; DataIngestion('1ShhXrwl89sZmwddLg1H8eHyW0IlmyKhY').download_data()"

# Step 2: Data Preprocessing
python -c "from src.sentiment.componet.data_preprocessing import DataPreprocessing; dp = DataPreprocessing(); df = dp.preprocess(); dp.save_preprocessed_data(df)"

# Step 3: Feature Engineering
python -c "from src.sentiment.componet.feature_engineering import FeatureEngineering; fe = FeatureEngineering(); df = fe.engineer_features(); fe.split_and_save(df)"

# Step 4: Model Training
python -c "from src.sentiment.componet.model_building import ModelTrainer; ModelTrainer().train()"

# Step 5: Entity Extraction
python -c "from src.entity.entity_extraction import EntityExtraction; ee = EntityExtraction(); df = ee.process_data(); ee.save_data(df)"

# Step 6: Recommendation System
python -c "from src.recommender.recommendation import RecommendationSystem; RecommendationSystem().train_and_save()"
```

### Pipeline Configuration

Edit `params.yaml` to customize model parameters:

```yaml
feature_engineering:
  test_size: 0.2          # Train-test split ratio
  random_state: 42        # Reproducibility seed

tfidf:
  max_features: 5000      # Maximum TF-IDF features
  ngram_range: [1, 2]     # Unigrams and bigrams

model:
  C: 0.1                  # SVM regularization
  max_iter: 5000          # Maximum iterations
  class_weight: balanced  # Handle class imbalance

sentiment_score:
  normalized_rating: 0.6  # Weight for avg rating
  positive_ratio: 0.4     # Weight for positive reviews
```

---

## 🌐 API Endpoints

### Start the API Server

```bash
# Local development
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

Access the API:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoint Details

#### 1. Root Endpoint
```http
GET /
```

**Response:**
```json
{
  "status": "success",
  "message": "HealthKart Recommendation System API is running"
}
```

---

#### 2. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "success",
  "message": "All systems operational"
}
```

**Status Codes:**
- `200`: Service healthy
- `503`: Models not loaded or service unavailable

---

#### 3. Sentiment Prediction
```http
POST /predict-sentiment
```

**Request Body:**
```json
{
  "text": "This product is amazing! I love it!"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 2.456,
  "text": "This product is amazing! I love it!"
}
```

**Sentiment Values:**
- `positive`: Rating >= 4
- `neutral`: Rating = 3
- `negative`: Rating < 3

**Status Codes:**
- `200`: Success
- `400`: Invalid input (empty text)
- `500`: Server error
- `503`: Model not loaded

---

#### 4. Product Recommendations
```http
POST /recommend
```

**Request Body:**
```json
{
  "product_name": "Pink Friday: Roman Reloaded Re-Up (w/dvd)",
  "n_recommendations": 10
}
```

**Response:**
```json
{
  "query_product": "Pink Friday: Roman Reloaded Re-Up (w/dvd)",
  "total_recommendations": 10,
  "recommendations": [
    {
      "product_name": "Similar Product 1",
      "brand": "Brand Name",
      "category": "Category Name",
      "avg_rating": 4.5,
      "hybrid_score": 0.856,
      "content_score": 0.923,
      "sentiment_score": 0.845,
      "collab_score": 0.678
    },
    ...
  ]
}
```

**Parameters:**
- `product_name`: Exact product name from dataset (required)
- `n_recommendations`: Number of recommendations (1-50, default: 10)

**Score Breakdown:**
- `hybrid_score`: Combined weighted score (final ranking)
- `content_score`: Content-based similarity (features, category, brand)
- `sentiment_score`: Aggregated review sentiment with confidence
- `collab_score`: Collaborative filtering (co-occurrence patterns)

**Status Codes:**
- `200`: Success
- `400`: Invalid input
- `404`: Product not found
- `500`: Server error
- `503`: Model not loaded

---

## Docker Deployment
### Docker Compose 

**Start services:**
```bash
# Build and start in foreground
docker-compose up --build

# Start in background (detached mode)
docker-compose up -d --build
```

**View logs:**
```bash
docker-compose logs -f
```

**Stop services:**
```bash
docker-compose down

# Remove volumes too
docker-compose down -v
```

**Rebuild after code changes:**
```bash
docker-compose up --build --force-recreate
```

---

### Push to Docker Hub

1. **Login to Docker Hub:**
```bash
docker login
```

2. **Tag the image:**
```bash
docker tag healthkart-recommendation:latest yourusername/healthkart-recommendation:latest
docker tag healthkart-recommendation:latest yourusername/healthkart-recommendation:v1.0
```

3. **Push to Docker Hub:**
```bash
docker push yourusername/healthkart-recommendation:latest
docker push yourusername/healthkart-recommendation:v1.0
```

4. **Pull and run from Docker Hub:**
```bash
docker pull yourusername/healthkart-recommendation:latest
docker run -p 8000:8000 yourusername/healthkart-recommendation:latest
```

---


## 💡 Usage Examples

### cURL Examples

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Sentiment Analysis:**
```bash
curl -X POST "http://localhost:8000/predict-sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing! I love it!"}'
```

**Get Recommendations:**
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Pink Friday: Roman Reloaded Re-Up (w/dvd)",
    "n_recommendations": 5
  }'
```

---

### Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Sentiment Analysis
response = requests.post(
    f"{BASE_URL}/predict-sentiment",
    json={"text": "This product is absolutely wonderful!"}
)
print(response.json())
# Output: {"sentiment": "positive", "confidence": 2.34, "text": "..."}

# Product Recommendations
response = requests.post(
    f"{BASE_URL}/recommend",
    json={
        "product_name": "Pink Friday: Roman Reloaded Re-Up (w/dvd)",
        "n_recommendations": 10
    }
)
recommendations = response.json()
print(f"Query: {recommendations['query_product']}")
print(f"Total: {recommendations['total_recommendations']}")
for rec in recommendations['recommendations'][:3]:
    print(f"- {rec['product_name']} (Score: {rec['hybrid_score']:.3f})")
```

---

### JavaScript/Fetch Example

```javascript
// Sentiment Analysis
fetch('http://localhost:8000/predict-sentiment', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'This product is great!' })
})
  .then(res => res.json())
  .then(data => console.log(data));

// Product Recommendations
fetch('http://localhost:8000/recommend', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    product_name: 'Pink Friday: Roman Reloaded Re-Up (w/dvd)',
    n_recommendations: 10
  })
})
  .then(res => res.json())
  .then(data => {
    console.log('Query:', data.query_product);
    data.recommendations.forEach(rec => {
      console.log(`${rec.product_name} - Score: ${rec.hybrid_score}`);
    });
  });
```

---

## 🧠 Model Architecture

### Sentiment Analysis Pipeline
```
Raw Text → Tokenization → Stopword Removal → Stemming → TF-IDF Vectorization → LinearSVC → Prediction
```

**Model Details:**
- **Algorithm**: Linear Support Vector Classification (LinearSVC)
- **Vectorization**: TF-IDF (max 5000 features, unigrams + bigrams)
- **Classes**: 3 (Negative, Neutral, Positive)
- **Class Weighting**: Balanced (handles imbalanced data)
- **Regularization**: C=0.1

**Performance Metrics:**
- Training accuracy: ~85-90%
- Handles imbalanced sentiment classes
- Confidence scores via decision function

---

### Recommendation System

**Hybrid Architecture:**
```
Input Product → [Content-Based (40%) + Sentiment Score (40%) + Collaborative (20%)] → Ranked Recommendations
```

#### 1. Content-Based Filtering (40%)
- TF-IDF vectorization of product features (brand, category, attributes)
- Cosine similarity between product vectors
- Captures feature-level similarity

#### 2. Sentiment Scoring (40%)
- Aggregates review ratings and positive sentiment ratio
- Confidence weighting based on review count
- Formula: `(avg_rating * 0.6 + positive_ratio * 0.4) * confidence`

#### 3. Collaborative Filtering (20%)
- Co-occurrence matrix from positive reviews
- Products reviewed together by same users/categories
- Category and brand-based associations

**Final Score Calculation:**
```
Hybrid Score = (0.4 × Content Score) + (0.4 × Sentiment Score) + (0.2 × Collab Score)
```

---

## 📊 Dataset Information

**Source**: Amazon Customer Reviews Dataset

**Key Columns:**
- `id`: Unique product identifier
- `name`: Product name
- `brand`: Brand name
- `categories`: Hierarchical category structure
- `reviews.text`: Customer review text
- `reviews.rating`: Rating (1-5 stars)
- `reviews.title`: Review title

**Statistics:**
- Total reviews: ~150,000+
- Products: ~20,000+
- Brands: ~3,000+
- Categories: ~500+

---

## 🔧 Configuration

### params.yaml Structure

```yaml
feature_engineering:
  test_size: 0.2                    # 80-20 train-test split
  random_state: 42                  # Reproducibility

tfidf:
  max_features: 5000                # TF-IDF vocabulary size
  ngram_range: [1, 2]               # Unigrams and bigrams

model:
  C: 0.1                            # SVM regularization parameter
  max_iter: 5000                    # Max training iterations
  class_weight: balanced            # Auto-balance class weights

sentiment_score:
  normalized_rating: 0.6            # Weight for average rating
  positive_ratio: 0.4               # Weight for positive review %
```

**Recommendation Weights** (in `recommendation.py`):
```python
sentiment_weight = 0.4      # Sentiment score importance
content_weight = 0.4        # Content similarity importance
collaborative_weight = 0.2  # Collaborative filtering importance
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Model not found error:**
```bash
# Solution: Run training pipeline first
python training_pipeline.py
```

**2. NLTK/SpaCy data not found:**
```bash
# Download NLTK data
python -m nltk.downloader punkt stopwords

# Download SpaCy model
python -m spacy download en_core_web_sm
```

**3. Port 8000 already in use:**
```bash
# Use different port
uvicorn app:app --port 8080

# Or kill process using port 8000
# Windows: netstat -ano | findstr :8000 → taskkill /PID <PID> /F
# Linux/Mac: lsof -ti:8000 | xargs kill -9
```

**4. Docker build fails:**
```bash
# Clear Docker cache and rebuild
docker system prune -a
docker-compose build --no-cache
```

**5. Out of memory during training:**
- Reduce `max_features` in `params.yaml`
- Use smaller dataset sample
- Increase system RAM or Docker memory limit

**6. Product not found in recommendations:**
- Check exact product name spelling
- Product must exist in training data
- Run entity extraction again if needed

---

## 📝 Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document functions with docstrings
- Keep functions modular and reusable

### Adding New Features

**1. New sentiment model:**
- Add to `src/sentiment/componet/model_building.py`
- Update inference in `src/sentiment/inference.py`
- Register in `app.py`

**2. New recommendation strategy:**
- Extend `RecommendationSystem` class
- Update score calculation in `getRecommendations()`
- Adjust weights in configuration

**3. New API endpoint:**
- Add route in `app.py`
- Create Pydantic models for request/response
- Update this README

### Testing

```bash
# Test sentiment prediction
python src/sentiment/inference.py

# Test recommendations
python src/recommender/inference.py

# Test API (after starting server)
curl http://localhost:8000/health
```


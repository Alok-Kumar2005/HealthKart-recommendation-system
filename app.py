from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.sentiment.inference import SentimentInference
from src.recommender.inference import RecommenderInference
from src.exception import CustomException
from src.logging import logging


app = FastAPI(
    title="HealthKart Recommendation System API",
    description="API for sentiment analysis and product recommendations",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### models
sentiment_model = None
recommender_model = None


@app.on_event("startup")
async def startup_event():
    global sentiment_model, recommender_model
    try:
        logging.info("Loading models...")
        sentiment_model = SentimentInference()
        recommender_model = RecommenderInference()
        logging.info("Models loaded successfully")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise e


### model schemas
class SentimentRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This product is amazing! I love it!"
            }
        }


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: Optional[float]
    text: str


class RecommendationRequest(BaseModel):
    product_name: str
    n_recommendations: int = 10

    class Config:
        json_schema_extra = {
            "example": {
                "product_name": "Pink Friday: Roman Reloaded Re-Up (w/dvd)",
                "n_recommendations": 10
            }
        }


class ProductRecommendation(BaseModel):
    product_name: str
    brand: str
    category: str
    avg_rating: float
    hybrid_score: float
    content_score: float
    sentiment_score: float
    collab_score: float
    trend_score: float 
    trend_direction: str


class RecommendationResponse(BaseModel):
    query_product: str
    recommendations: List[ProductRecommendation]
    total_recommendations: int


class HealthResponse(BaseModel):
    status: str
    message: str


### api endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    return {
        "status": "success",
        "message": "HealthKart Recommendation System API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        if sentiment_model is None or recommender_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Check if temporal analysis is available
        has_temporal = recommender_model.trend_scores is not None
        
        return {
            "status": "success",
            "message": f"All systems operational (Temporal Analysis: {'Enabled' if has_temporal else 'Disabled'})"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))



@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        if recommender_model is None:
            raise HTTPException(status_code=503, detail="Recommender model not loaded")
        
        if not request.product_name or not request.product_name.strip():
            raise HTTPException(status_code=400, detail="Product name cannot be empty")
        
        if request.n_recommendations < 1 or request.n_recommendations > 50:
            raise HTTPException(
                status_code=400, 
                detail="Number of recommendations must be between 1 and 50"
            )
        
        recommendations_df = recommender_model.recommend(
            request.product_name,
            n_recommendations=request.n_recommendations
        )
        
        if isinstance(recommendations_df, str):
            raise HTTPException(status_code=404, detail=recommendations_df)
        
        recommendations = recommendations_df.to_dict('records')
        
        return {
            "query_product": request.product_name,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations)
        }
        
    except HTTPException:
        raise
    except CustomException as e:
        logging.error(f"Custom exception in recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Error in recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
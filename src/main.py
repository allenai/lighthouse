"""Coastal Detection Service"""

from __future__ import annotations

import logging.config
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from pipeline import land_water_mapping
from pipeline import main as pipeline_main

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
HOST: str = "0.0.0.0"  # nosec B104
PORT: Union[str, int] = os.getenv("COASTAL_DETECTION_PORT", 8000)
MODEL_VERSION: Union[str, datetime] = os.getenv("GIT_COMMIT_HASH", datetime.today())

app = FastAPI()
logger.info("Starting Coastal Detection Service")


class RoundedFloat(float):
    """Rounds floats to a specified number of decimal places."""

    def __new__(
        cls, value: Union[float, np.float32, np.float64], n_decimals: Optional[int] = 5
    ) -> RoundedFloat:
        if n_decimals is not None:
            cls.n_decimals = n_decimals
        return super().__new__(cls, round(float(value), cls.n_decimals))

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Define custom JSON schema for RoundedFloat."""
        return {
            "type": "number",
            "description": "A float rounded to a specified precision",
        }

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: Any) -> Any:
        """Pydantic v2 core schema compatibility."""

        def validate(value: Union[float, np.float32, np.float64]) -> RoundedFloat:
            if isinstance(value, (np.float32, np.float64)):
                value = float(value)
            return cls(value)

        return validate


class CoastalDetectionRequest(BaseModel):
    """Request object for coastal detections."""

    lat: float = Field(..., description="Latitude of the point")
    lon: float = Field(..., description="Longitude of the point")

    class Config:
        """Example configuration for a request."""

        json_schema_extra = {
            "example": {
                "lat": 47.636895,
                "lon": -122.334984,
            },
        }


class CoastalDetectionResponse(BaseModel):
    """Response object for coastal detections."""

    distance_to_coast_m: int
    land_cover_class: str
    nearest_coastal_point: List[float]
    version: datetime


@app.on_event("startup")
async def initialize() -> None:
    """Initialize resources such as loading models or BallTree."""
    logger.info("Initializing global resources for Coastal Detection Service")


@app.get("/")
async def home() -> Dict[str, str]:
    """Health check endpoint to confirm the service is running."""
    return {"message": "Coastal Detection Service is running"}


@app.post("/detect", response_model=CoastalDetectionResponse)
async def detect_coastal_info(
    request: CoastalDetectionRequest, response: Response
) -> CoastalDetectionResponse:
    """
    Detect coastal information for given coordinates.

    Args:
        request: The request containing lat/lon coordinates
        response: FastAPI response object

    Returns:
        CoastalDetectionResponse with distance, land cover, and nearest point

    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        # Run the detection logic from pipeline
        distance_m, land_class_id, nearest_point = pipeline_main(
            request.lat, request.lon
        )

        # Map land cover class
        land_cover_class = land_water_mapping.get(land_class_id, "Unknown")
        logger.info(land_cover_class)
        logger.info(distance_m)

        # Prepare the response with rounded coordinates
        return CoastalDetectionResponse(
            distance_to_coast_m=int(distance_m),
            land_cover_class=land_cover_class,
            nearest_coastal_point=[
                round(float(nearest_point[0]), 5),
                round(float(nearest_point[1]), 5),
            ],
            version=datetime.today(),
        )

    except Exception as e:
        logger.error(f"Error in processing detection request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=int(PORT), proxy_headers=True)

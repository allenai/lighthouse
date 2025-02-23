"""Coastal Detection Service"""

from __future__ import annotations

import logging.config
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Dict, List, Union

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from pipeline import land_water_mapping
from pipeline import main as pipeline_main

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
HOST: str = "0.0.0.0"  # nosec B104
PORT: Union[str, int] = os.getenv("COASTAL_DETECTION_PORT", 8000)
MODEL_VERSION: Union[str, datetime] = os.getenv("GIT_COMMIT_HASH", datetime.today())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle startup and shutdown events."""
    # Startup
    logger.info("Starting Coastal Detection Service")
    yield
    # Shutdown
    logger.info("Shutting down Coastal Detection Service")


app = FastAPI(lifespan=lifespan)


class CoastalRequest(BaseModel):
    """Request object for coastal detections.

    If batch_mode is False, lat and lon should be single floats.
    If batch_mode is True, lat and lon should be lists of floats of equal length.
    """

    model_config = ConfigDict(strict=True)

    batch_mode: bool = Field(
        default=False,
        description="Enable batch mode for multiple coordinates.",
    )
    lat: Union[float, List[float]] = Field(
        ..., description="Latitude(s) of the point(s)"
    )
    lon: Union[float, List[float]] = Field(
        ..., description="Longitude(s) of the point(s)"
    )

    @model_validator(mode="after")
    def check_lat_lon(
        self,
        info: ValidationInfo,
    ) -> "CoastalRequest":
        """Validate that lat/lon match the batch_mode.

        Args:
            info: Validation context information

        Returns:
            The validated model instance

        Raises:
            ValueError: If lat/lon types don't match batch_mode setting
        """
        if self.batch_mode:
            # batch_mode = True, lat and lon must be lists
            if not isinstance(self.lat, list) or not isinstance(self.lon, list):
                raise ValueError("lat and lon must be lists when batch_mode is True")
            if len(self.lat) != len(self.lon):
                raise ValueError(
                    "lat and lon must have the same length when batch_mode is True"
                )
        else:
            # batch_mode = False, lat and lon must be single floats
            if isinstance(self.lat, list) or isinstance(self.lon, list):
                raise ValueError(
                    "lat and lon must be single floats when batch_mode is False"
                )
        return self


class CoastalDetectionResponse(BaseModel):
    """Response object for a single coastal detection result."""

    distance_to_coast_m: int
    land_cover_class: str
    nearest_coastal_point: List[float]
    version: datetime


@app.get("/")
async def home() -> Dict[str, str]:
    """Health check endpoint to confirm the service is running."""
    return {"message": "Coastal Detection Service is running"}


@app.post("/detect")
async def detect_coastal_info(
    request: CoastalRequest,
    response: Response,
) -> Union[CoastalDetectionResponse, List[CoastalDetectionResponse]]:
    """Detect coastal information for given coordinates.

    Args:
        request: The request containing lat/lon coordinates and batch_mode
        response: FastAPI response object

    Returns:
        - If batch_mode=False: A single CoastalDetectionResponse
        - If batch_mode=True: A list of CoastalDetectionResponse objects

    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        if not request.batch_mode:
            # Single coordinate processing
            distance_m, land_class_id, nearest_point = pipeline_main(
                request.lat, request.lon, batch_mode=False
            )

            # Map land cover class
            land_cover_class = land_water_mapping.get(land_class_id, "Unknown")
            logger.info("Land cover class: %s", land_cover_class)
            logger.info("Distance to coast: %d m", distance_m)

            return CoastalDetectionResponse(
                distance_to_coast_m=int(distance_m),
                land_cover_class=land_cover_class,
                nearest_coastal_point=[
                    round(float(nearest_point[0]), 5),
                    round(float(nearest_point[1]), 5),
                ],
                version=datetime.today(),
            )
        else:
            # Batch mode: lat and lon are lists
            lats = np.array(request.lat)
            lons = np.array(request.lon)
            df_results = pipeline_main(lats, lons, batch_mode=True)

            responses = []
            for i, row in df_results.iterrows():
                land_cover_class = land_water_mapping.get(
                    int(row["land_class"]), "Unknown"
                )
                responses.append(
                    CoastalDetectionResponse(
                        distance_to_coast_m=int(row["distance_m"]),
                        land_cover_class=land_cover_class,
                        nearest_coastal_point=[
                            round(float(row["nearest_lat"]), 5),
                            round(float(row["nearest_lon"]), 5),
                        ],
                        version=datetime.today(),
                    )
                )
            return responses

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error("Error in processing request: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=int(PORT),
        proxy_headers=True,
        workers=25,  # Add this line
    )

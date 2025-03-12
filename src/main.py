from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, model_validator, ValidationError

import numpy as np
import logging
import os
from datetime import datetime
from typing import List, Union

from pipeline import land_water_mapping
from pipeline import main as pipeline_main

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
HOST = "0.0.0.0"  # nosec B104
PORT = os.getenv("COASTAL_DETECTION_PORT", 8000)
MODEL_VERSION = os.getenv("GIT_COMMIT_HASH", datetime.today())

app = FastAPI()


@app.get("/")
async def home() -> dict:
    """Health check endpoint to confirm the service is running."""
    return {"message": "Coastal Detection Service is running"}


class CoastalRequest(BaseModel):
    """Request object for coastal detections."""

    batch_mode: bool = Field(
        default=False, description="Enable batch mode for multiple coordinates."
    )
    lat: Union[float, List[float]] = Field(
        ..., description="Latitude(s) of the point(s)"
    )
    lon: Union[float, List[float]] = Field(
        ..., description="Longitude(s) of the point(s)"
    )

    @model_validator(mode="after")
    def check_lat_lon(self) -> "CoastalRequest":
        """Validate lat/lon values."""
        if self.batch_mode:
            if not isinstance(self.lat, list) or not isinstance(self.lon, list):
                raise ValueError("lat and lon must be lists when batch_mode is True")
            if len(self.lat) != len(self.lon):
                raise ValueError(
                    "lat and lon must have the same length when batch_mode is True"
                )

            # Validate each latitude & longitude
            for lat, lon in zip(self.lat, self.lon):
                if not (-90 <= lat <= 90):
                    raise ValueError(
                        f"Invalid latitude: {lat}. Must be between -90 and 90."
                    )
                if not (-180 <= lon <= 180):
                    raise ValueError(
                        f"Invalid longitude: {lon}. Must be between -180 and 180."
                    )

        else:
            if isinstance(self.lat, list) or isinstance(self.lon, list):
                raise ValueError(
                    "lat and lon must be single floats when batch_mode is False"
                )

            # Validate single latitude & longitude
            if not (-90 <= self.lat <= 90):
                raise ValueError(
                    f"Invalid latitude: {self.lat}. Must be between -90 and 90."
                )
            if not (-180 <= self.lon <= 180):
                raise ValueError(
                    f"Invalid longitude: {self.lon}. Must be between -180 and 180."
                )

        return self


class CoastalDetectionResponse(BaseModel):
    """Response object for a single coastal detection result."""

    distance_to_coast_m: int
    land_cover_class: str
    nearest_coastal_point: List[float]
    version: datetime


@app.post("/detect")
async def detect_coastal_info(
    request: CoastalRequest, response: Response
) -> Union[CoastalDetectionResponse, List[CoastalDetectionResponse]]:
    """Detect coastal information for given coordinates."""
    try:
        if not request.batch_mode:
            # Single coordinate processing
            distance_m, land_class_id, nearest_point = pipeline_main(
                request.lat, request.lon, batch_mode=False
            )

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
            # Batch mode
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

    except ValidationError as e:
        logger.error("Validation error: %s", e)
        raise HTTPException(
            status_code=422, detail=str(e)
        )  

    except ValueError as e:
        logger.error("User input error: %s", e)
        raise HTTPException(
            status_code=400, detail=str(e)
        ) 

    except Exception as e:
        logger.error("Internal server error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=int(PORT), proxy_headers=True, workers=25)

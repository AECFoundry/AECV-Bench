"""
Data models for floor plan element extraction.
"""
from pydantic import BaseModel, Field


class PlanElements(BaseModel):
    """Schema for floor plan element counts."""
    Door: int = Field(..., description="TOTAL number of doors (all types)")
    Window: int = Field(..., description="TOTAL number of windows (all types)")
    Space: int = Field(
        ...,
        description=(
            "TOTAL count of distinct spaces/rooms — "
            "include every enclosed area, large or small, habitable or not "
            "(bedrooms, closets, terrace, WC, storage, halls, etc.)."
        ),
    )
    Bedroom: int = Field(..., description="TOTAL number of bedrooms")
    Toilet: int = Field(..., description="TOTAL number of toilets/WCs")


def get_json_schema():
    """Get the JSON schema for PlanElements."""
    schema = PlanElements.model_json_schema()
    schema["additionalProperties"] = False  # Required by OpenAI response_format
    return schema


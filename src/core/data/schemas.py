"""
Pydantic Schemas for Structured Document
"""

from pydantic import BaseModel, Field
from typing import Optional


class ShipmentDetailsModel(BaseModel):
    """
    Mandatory canonical shipment fields.
    All fields are nullable to handle non-logistic documents.
    """

    Shipment_id: Optional[str] = Field(None, description="Unique shipment identifier")
    shipper: Optional[str] = Field(None, description="Sender of the goods")
    consignee: Optional[str] = Field(None, description="Receiver of the goods")
    pickup_datetime: Optional[str] = Field(
        None, description="Planned pickup date and time"
    )
    delivery_datetime: Optional[str] = Field(
        None, description="Planned delivery date and time"
    )
    equipment_type: Optional[str] = Field(
        None, description="Type of vehicle or container"
    )
    mode: Optional[str] = Field(
        None, description="Transportation mode (e.g., FTL, LTL)"
    )
    rate: Optional[str] = Field(None, description="Agreed freight rate")
    currency: Optional[str] = Field(None, description="Currency of the rate")
    weight: Optional[str] = Field(None, description="Total weight of the shipment")
    carrier_name: Optional[str] = Field(
        None, description="Name of the transport company"
    )


class StructuredDocumentModel(BaseModel):
    """
    Represents full structured document.
    """

    shipment_details: ShipmentDetailsModel

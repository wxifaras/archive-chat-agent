from pydantic import BaseModel, Field, RootModel, HttpUrl
from datetime import date, datetime
from typing import List, Optional

class EmailItem(BaseModel):
    text: Optional[str] = None
    projectId: Optional[int] = None
    Author: Optional[str] = None
    Email_Subject: Optional[str] = Field(default=None, alias='Email Subject')
    Received_Date: Optional[date] = Field(default=None, alias='Received Date')
    Key_Topics: Optional[str] = Field(default=None, alias='Key Topics')
    Email_body: Optional[str] = Field(default=None, alias='Email body')
    Provenance: Optional[str] = None
    Email_ID: Optional[str] = Field(default=None, alias='Email ID')
    URL_Index: Optional[str] = Field(default=None, alias='URL Index')
    URL_Type: Optional[str] = Field(default=None, alias='URL Type')
    Force_Scraper: Optional[str] = Field(default=None, alias='Force Scraper')
    crawledLink: Optional[HttpUrl] = None
    links: Optional[List[str]] = None
    allLinks: Optional[List[str]] = None
    level: Optional[int] = None
    status: Optional[str] = None
    createdOn: Optional[datetime] = None
    jobDomain: Optional[HttpUrl] = None
    Source: Optional[str] = None
    Client_Exposure: Optional[int] = Field(default=None, alias='Client Exposure')
    POV_Rating: Optional[int] = Field(default=None, alias='POV Rating')
    Comments: Optional[str] = None
    Timestamp: Optional[str] = None
    Provenance_Source: Optional[str] = None

    model_config = {
        'populate_by_name': True,
    }

class EmailList(RootModel[List[EmailItem]]):
    """
    Wraps the top-level JSON array of EmailItem objects.
    Supports parsing directly from List[dict].
    """
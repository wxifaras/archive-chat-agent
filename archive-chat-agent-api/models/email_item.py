from pydantic import BaseModel, Field, RootModel, HttpUrl
from datetime import date, datetime
from typing import List

class EmailItem(BaseModel):
    text: str
    projectId: int
    Author: str
    Email_Subject: str = Field(alias='Email Subject')
    Received_Date: date = Field(alias='Received Date')
    Key_Topics: str = Field(alias='Key Topics')
    Email_body: str = Field(alias='Email body')
    Provenance: str
    Email_ID: str = Field(alias='Email ID')
    URL_Index: str = Field(alias='URL Index')
    URL_Type: str = Field(alias='URL Type')
    Force_Scraper: str = Field(alias='Force Scraper')
    crawledLink: HttpUrl
    links: List[str]
    allLinks: List[str]
    level: int
    status: str
    createdOn: datetime
    jobDomain: HttpUrl

    model_config = {
        'populate_by_name': True,
    }

class EmailList(RootModel[List[EmailItem]]):
    """
    Wraps the top-level JSON array of EmailItem objects.
    Supports parsing directly from List[dict].
    """
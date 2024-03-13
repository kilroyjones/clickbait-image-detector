from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class QueryParsed(BaseModel):
    publishedAt: str
    channelId: str
    title: str
    description: str
    thumbnails: Thumbnails
    channelTitle: str
    liveBroadcastContent: str
    publishTime: str


"""
Data from query to YouTube API which is used. 
"""

from pydantic import BaseModel
from app.models.youtube_api_result import Thumbnails 

class QueryParsed(BaseModel):
    """
    Used as a model to parse the response from YouTube's API
    """

    publishedAt: str
    channelId: str
    title: str
    description: str
    thumbnails: Thumbnails
    channelTitle: str
    liveBroadcastContent: str
    publishTime: str


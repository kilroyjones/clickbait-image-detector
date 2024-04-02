from pydantic import BaseModel
from models.youtube_api_result import Thumbnails 

class QueryParsed(BaseModel):
    publishedAt: str
    channelId: str
    title: str
    description: str
    thumbnails: Thumbnails
    channelTitle: str
    liveBroadcastContent: str
    publishTime: str


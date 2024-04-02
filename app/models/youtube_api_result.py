"""
Complete response fom your YouTube API
"""

from typing import List, Optional
from pydantic import BaseModel

class Thumbnail(BaseModel):
    url: str
    width: int
    height: int

class Thumbnails(BaseModel):
    default: Thumbnail
    medium: Thumbnail
    high: Thumbnail

class VideoSnippet(BaseModel):
    publishedAt: str
    channelId: str
    title: str
    description: str
    thumbnails: Thumbnails
    channelTitle: str
    liveBroadcastContent: str
    publishTime: str

class VideoId(BaseModel):
    kind: str
    videoId: str

class Video(BaseModel):
    kind: str
    etag: str
    id: VideoId
    snippet: VideoSnippet

class PageInfo(BaseModel):
    totalResults: int
    resultsPerPage: int

class YouTubeAPIResult(BaseModel):
    kind: str
    etag: str
    nextPageToken: Optional[str] = None
    regionCode: str
    pageInfo: PageInfo
    items: List[Video]

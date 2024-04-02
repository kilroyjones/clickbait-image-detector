"""
YouTube API module

This is used all the YouTube API and get back video details. 

"""

from typing import List, Dict
from googleapiclient.discovery import build
from app.models.youtube_api_result import YouTubeAPIResult


def search(query: str, api_key: str, max_results: int = 10):
    """ 
    Performs a search on YouTube and returns a list of video data including titles, descriptions, video IDs, and thumbnail URLs.

    Args:
        query (str): The search query.
        api_key (str): The API key for accessing YouTube Data API.
        max_results (int, optional): Maximum number of results to return. Defaults to 10.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing data about a video.
    """


    youtube  = build('youtube', 'v3', developerKey=api_key)

    # pylint: disable=no-member
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results
    )

    response = request.execute()
    query_result: YouTubeAPIResult = YouTubeAPIResult.model_validate(response)

    results: List[Dict[str, str]] = []
    if query_result:
        for item in query_result.items:
            data: Dict[str, str] = {
                "query": query,
                "title": item.snippet.title,
                "description": item.snippet.description,
                "video_id": item.id.videoId,
                "thumbnail_url": item.snippet.thumbnails.default.url,
            }
            results.append(data)

    return results



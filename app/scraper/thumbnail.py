"""
TODO: Add this as an option for the scrape
"""
from typing import Tuple

import requests


def get_youtube_thumbnail_url(video_id: str) -> str:
    """
    Generates the URL for a YouTube video's highest quality thumbnail.

    Parameters:
    - video_id: str, the unique ID of the YouTube video.

    Returns:
    - str, URL of the highest quality thumbnail image.
    """
    base_url: str = "https://img.youtube.com/vi/"
    highest_quality: str = "/maxresdefault.jpg"
    
    thumbnail_url: str = f"{base_url}{video_id}{highest_quality}"
    return thumbnail_url


def fetch_thumbnail(url: str) -> Tuple[bytes, bool]:
    """
    Fetches the thumbnail image from a given URL.

    Parameters:
    - url: str, the URL of the image to fetch.

    Returns:
    - Tuple[bytes, bool]: The content of the image in bytes and a boolean indicating success.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad responses
        return (response.content, True)
    except requests.RequestException:
        return (b'', False)
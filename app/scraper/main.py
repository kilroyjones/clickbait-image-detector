"""

"""

import sqlite3
import sys
import time
import warnings

import requests

import app.database.database as db

from app.models.video import Video


def download_images(conn: sqlite3.Connection, delay: int = 3):
    """
    Downloads images from a list of URLs, saving them by their YouTube ID.

    Args:
        image_urls (List[str]): The list of image URLs to download.
        delay (int): The number of seconds to wait between downloads to avoid hammering the server.
    """
    image_urls = db.get_downloads(conn)
    videos = [Video(**video_data) for video_data in image_urls]

    # Loop over all video information 
    for video in videos :
        try:
            response = requests.get(video.thumbnail_url, timeout=10)
            response.raise_for_status()  

            filename = f"{video.video_id}.jpg"

            with open('./downloads/' + filename, 'wb') as file:
                file.write(response.content)

            print(f"Downloaded {filename}")
            
            db.update_downloaded_status(conn, video.video_id)
            time.sleep(delay)
        except requests.RequestException as e:
            print(f"Failed to download {video.thumbnail_url}: {e}")


def main():
    """
    Main function 
    """
    
    # Set up database
    conn = db.create_connection('output.sqlite')
    if(conn):
        db.create_table(conn)
    else:
        warnings.warn("Unable to connection to database", UserWarning)
        sys.exit(1)  

    download_images(conn)
    
if __name__ == "__main__":
    main()

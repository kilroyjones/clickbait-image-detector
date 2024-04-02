"""
Query is used to get video information from YouTube

"""

import os
import argparse
import sqlite3
import sys
import time
import warnings

from typing import Final
from dotenv import load_dotenv

import app.database.database as db
import app.query.keywords as keywords
import app.query.api as api


def add_keywords(filename: str, api_key: str, conn: sqlite3.Connection):  
    """
    Pulls a single keyword from the file pass as an argument and the use the 
    YouTube API to get a video listing for it. 
    """

    keyword = keywords.get_keyword(filename) 

    while keyword is not None:
        print("Retrieving:", keyword)
        results = api.search(keyword, api_key)
        if(results):
            db.insert_multiple_videos(conn, results)
        keyword = keywords.get_keyword(filename)
        time.sleep(3)

def main():
    """
    This will pull keywords from a file given with the "-k" or "-keywords" parameter. 

    """

    load_dotenv()
    api_key: Final = os.getenv("YOUTUBE_APIV3")

    if not api_key:
        warnings.warn("YOUTUBE_APIV3 environment variable is missing.", UserWarning)
        sys.exit(1)
    
    # Set up database
    conn = db.create_connection('output.sqlite')
    if(conn):
        db.create_table(conn)
    else:
        warnings.warn("Unable to connection to database", UserWarning)
        sys.exit(1)  
    
    # Creates table if it doesn't already exist
    db.create_table(conn)

    parser = argparse.ArgumentParser(description="YouTube data scraper")
    parser.add_argument("-k", "--keywords", help="Input file of keywords to search", required=False)
    args = parser.parse_args()

    if args.keywords:
        add_keywords(args.keywords, api_key, conn)
    else:
        warnings.warn("File of keywords must be provided", UserWarning)
        sys.exit(1)  

if __name__ == "__main__":
    main()

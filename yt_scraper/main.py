# Libraries
import os
import sys
import warnings
import argparse

from typing import Final
from dotenv import load_dotenv

# Modules
import yt_scraper.keywords as keywords
import yt_scraper.yt_api as yt_api
import yt_scraper.database as db

# Startup
load_dotenv()



def main(): 
    """
    Main function 
    """

    api_key: Final = os.getenv("YOUTUBE_APIV3")
    if not api_key:
        warnings.warn("YOUTUBE_APIV3 environment variable is missing.", UserWarning)
        sys.exit(1)  # Exit the program with an error status (1).
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="YouTube data scraper")
    parser.add_argument("-k", "--keywords", help="Input file of keywords to search", required=True)

    args = parser.parse_args()

    # Check for argument
    if args.keywords:
        print(f"File: {args.keywords}")
    else:
        warnings.warn("File of keywords must be provided", UserWarning)
        sys.exit(1)  # Exit the program with an error status (1).
    
    conn = db.create_connection('output.sqlite')
    if(conn):
        db.create_table(conn)
    else:
        warnings.warn("Unable to connection to database", UserWarning)
        sys.exit(1)  # Exit the program with an error status (1).


    keyword = keywords.get_keyword(args.keywords) 
    while keyword is not None:
        results = yt_api.search(keyword, api_key)
        if(results):
            db.insert_multiple_videos(conn, results)

        keyword = keywords.get_keyword(args.keywords)

if __name__ == "__main__":
    main()

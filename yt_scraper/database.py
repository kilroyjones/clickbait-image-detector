import sqlite3
from typing import Dict, List


def create_connection(db_file: str) -> sqlite3.Connection:
    """Create a database connection to a SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to SQLite database: {db_file}")
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn: sqlite3.Connection):
    """Create a table for storing video data."""
    try:
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS videos (
                            id TEXT PRIMARY KEY,
                            query TEXT,
                            title TEXT,
                            description TEXT,
                            thumbnail_url TEXT
                          );""")
        print("Table created successfully.")
    except sqlite3.Error as e:
        print(e)

def insert_video_data(conn: sqlite3.Connection, video_data: Dict):
    """Insert video data into the videos table."""
    sql = '''INSERT INTO videos(id, query, title, description, thumbnail_url)
             VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
             query=excluded.query,
             title=excluded.title,
             description=excluded.description,
             thumbnail_url=excluded.thumbnail_url;'''
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (
            video_data['video_id'],
            video_data['query'],
            video_data['title'],
            video_data['description'],
            video_data['thumbnail_url']))
        conn.commit()
        print(video_data["video_id"] + " inserted")
    except sqlite3.Error as e:
        print(e)

def insert_multiple_videos(conn: sqlite3.Connection, videos: List[Dict]):
    """Insert multiple videos into the database."""
    for video in videos:
        insert_video_data(conn, video)

# # Example usage
# if __name__ == "__main__":
#     database = "youtube_videos.db"
#     conn = create_connection(database)
    
#     if conn is not None:
#         create_table(conn)
        
#         # Example video data from API call
#         videos_data = [
#             {"video_id": "abc123", "title": "Funny Cats", "description": "A video about funny cats.", "thumbnail_url": "http://example.com/cat1.jpg"},
#             {"video_id": "def456", "title": "Cute Dogs", "description": "A video about cute dogs.", "thumbnail_url": "http://example.com/dog1.jpg"},
#         ]
        
#         insert_multiple_videos(conn, videos_data)
        
#         conn.close()

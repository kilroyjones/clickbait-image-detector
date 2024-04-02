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

def get_all_videos(conn: sqlite3.Connection) -> List[Dict]:
    """Retrieve videos where downloaded is 0 or not defined."""
    videos = []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, class FROM videos WHERE class IS NOT NULL;")
        rows = cursor.fetchall()
        for row in rows:
            videos.append({
                "video_id": row[0],
                "cls": row[1]
            })
    except sqlite3.Error as e:
        print(e)
    return videos

def get_videos(conn: sqlite3.Connection, class_value: str) -> List[Dict]:
    """Retrieve videos where downloaded is 0 or not defined."""
    videos = []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, class FROM videos WHERE class=" + class_value + ";")
        rows = cursor.fetchall()
        for row in rows:
            videos.append({
                "video_id": row[0],
                "cls": row[1]
            })
    except sqlite3.Error as e:
        print(e)
    return videos

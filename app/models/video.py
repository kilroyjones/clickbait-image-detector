class Video:
    def __init__(self, video_id, query, title, description, thumbnail_url, downloaded=False):
        self.video_id = video_id
        self.query = query
        self.title = title
        self.description = description
        self.thumbnail_url = thumbnail_url
        self.downloaded = downloaded
    
    def __repr__(self):
        return f"Video(video_id={self.video_id}, title={self.title})"
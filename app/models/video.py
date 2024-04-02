"""
Models for use with database
"""

class Video:
    """
    Complete model from database
    """

    def __init__(self, video_id, query, title, description, thumbnail_url, downloaded=False, _class=None):
        self.video_id = video_id
        self.query = query
        self.title = title
        self.description = description
        self.thumbnail_url = thumbnail_url
        self.downloaded = downloaded
        self._class = _class


class VideoIdAndClass:
    """
    Model used here only to pull just ids and the class.
    """

    def __init__(self, video_id, cls):
        self.video_id = video_id
        self.cls = cls
    
"""

"""
import os
import sqlite3
import tkinter as tk
from PIL import Image, ImageTk

class ImageClassifier:
    """

    """
    def __init__(self, conn):
        self.conn = conn
        self.root = tk.Tk()
        self.panel = None
        self.image_iterator = self.get_image_iterator()
        self.image_id, self.image_path = next(self.image_iterator, (None, None))
        self.setup_ui()

    def get_image_iterator(self):
        """

        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM videos WHERE class IS NULL")
        rows = cursor.fetchall()
        return ((row[0], os.path.join("./downloads", f"{row[0]}.jpg")) for row in rows if os.path.exists(os.path.join("./downloads", f"{row[0]}.jpg")))

    def update_class(self, image_id, class_id):
        """

        """
        cursor = self.conn.cursor()
        cursor.execute("UPDATE videos SET class = ? WHERE id = ?", (class_id, image_id))
        self.conn.commit()

    def on_key_press(self, event):
        """
        
        """
        if event.keysym == 'Escape':
            self.root.destroy()
            self.conn.close()
            exit(0)
        else:
            try:
                class_id = int(event.char)
                self.update_class(self.image_id, class_id)
                self.image_id, self.image_path = next(self.image_iterator, (None, None))
                if self.image_path:
                    img = Image.open(self.image_path)
                    imgTk = ImageTk.PhotoImage(img)
                    self.panel.configure(image=imgTk)
                    self.panel.image = imgTk
                else:
                    self.root.destroy()
                    self.conn.close()
            except ValueError:
                pass

    def setup_ui(self):
        """
        
        """
        self.root.bind("<Key>", self.on_key_press)
        if self.image_path:
            img = Image.open(self.image_path)
            imgTk = ImageTk.PhotoImage(img)
            self.panel = tk.Label(self.root, image=imgTk)
            self.panel.image = imgTk
            self.panel.pack(side="bottom", fill="both", expand="yes")
        else:
            print("No images to classify.")
            self.root.destroy()

def main():
    """
    
    """
    conn = sqlite3.connect('output.sqlite')
    app = ImageClassifier(conn)
    app.root.mainloop()

    conn.close()

if __name__ == "__main__":
    main()

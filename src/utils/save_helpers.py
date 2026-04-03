import os

def make_unique_filename(path):
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 1

    while True:
        new_path = f"{base}-{counter:03d}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1
import os
import requests

def download_model(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}")
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)
        print(f"Saved {filename}")

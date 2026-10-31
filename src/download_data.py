# Download the full Dataset and pretrained GloVe vectors into data/

import zipfile
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

NEWS_URL = "https://huggingface.co/datasets/heegyu/news-category-dataset/resolve/main/data.json"
NEWS_FILE = DATA_DIR / "News_Category_Dataset_v3.json"

GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_MEMBER = "glove.6B.100d.txt"
GLOVE_FILE = DATA_DIR / GLOVE_MEMBER


def download(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"  {dest.name}: {100 * done / total:5.1f}%", end="\r")
    print()
    tmp.replace(dest)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    if NEWS_FILE.exists():
        print(f"{NEWS_FILE.name} already downloaded")
    else:
        print("Downloading News Category Dataset")
        download(NEWS_URL, NEWS_FILE)

    if GLOVE_FILE.exists():
        print(f"{GLOVE_FILE.name} already downloaded")
    else:
        print("Downloading GloVe 6B")
        zip_path = DATA_DIR / "glove.6B.zip"
        download(GLOVE_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extract(GLOVE_MEMBER, DATA_DIR)
        zip_path.unlink()

    print("Finshed")


if __name__ == "__main__":
    main()

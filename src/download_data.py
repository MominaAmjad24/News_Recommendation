import os
import zipfile
import urllib.request

def download_and_extract(url, extract_to):
    os.makedirs("data", exist_ok=True)
    zip_path = os.path.join("data", url.split("/")[-1])

    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)

    print("Done.\n")

if __name__ == "__main__":
    train_url = "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
    dev_url = "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"

    download_and_extract(train_url, "data/MINDsmall_train")
    download_and_extract(dev_url, "data/MINDsmall_dev")

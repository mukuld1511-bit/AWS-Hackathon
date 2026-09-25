import urllib.request
import zipfile
import os

url = "https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip"
zip_path = "student_resource.zip"

print(f"Downloading {url}...")
urllib.request.urlretrieve(url, zip_path)
print("Download complete. Extracting...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(".")

print("Extraction complete. Cleaning up...")
os.remove(zip_path)
print("Done!")

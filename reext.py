import os, zipfile

DATA_DIR = "data/"
data_dirpath = os.path.abspath(DATA_DIR)

print("Re-extracting all zip files in data/ ...")

for filename in os.listdir(data_dirpath):
    if filename.endswith(".zip"):
        zip_path = os.path.join(data_dirpath, filename)
        print(f"Extracting from {filename} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dirpath)

print("\n✅ Extraction complete. All zip contents refreshed in data/.\n")

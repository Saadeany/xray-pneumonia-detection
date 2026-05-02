import os
import zipfile

def download_dataset():
    print("Downloading dataset from Kaggle...")
    os.system("kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")

    print("Extracting dataset...")
    with zipfile.ZipFile("chest-xray-pneumonia.zip", 'r') as zip_ref:
        zip_ref.extractall("data")

    print("Dataset ready at data/chest_xray/")

if __name__ == "__main__":
    download_dataset()
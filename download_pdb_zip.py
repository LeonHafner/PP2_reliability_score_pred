#pip install gdown
import gdown
import zipfile
import os

def download_and_unzip_google_drive(file_url, output_dir="all_pdb_files"):

    file_id = file_url.split("/d/")[1].split("/view")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "downloaded_file.zip")

    print("Downloading the file...")
    gdown.download(download_url, output_file, quiet=False)

    if zipfile.is_zipfile(output_file):
        print("Unzipping the file...")
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Files extracted to {output_dir}")
    else:
        print("Downloaded file is not a valid zip file.")

    os.remove(output_file)

file_url = "https://drive.google.com/file/d/1W2Wc5jpbNe0dqC8sFwbjeEJL9D8Suet3/view?usp=sharing"

download_and_unzip_google_drive(file_url)

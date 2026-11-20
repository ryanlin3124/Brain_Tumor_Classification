import zipfile
from pathlib import Path
import os

data_path = Path("C:\Ryan Lin Files")
image_path = data_path / "Brain_Tumor_Python" / "Epic and CSCR hospital Dataset"

if image_path.is_dir():
    print(f"{image_path} directory exists")
else:
    print("DNE")
    image_path.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(data_path / "Brain_Tumor_Python" / "Epic and CSCR hospital Dataset.zip", "r") as zip_ref:
    zip_ref.extractall(image_path)

os.remove(data_path / "Brain_Tumor_Python" / "Epic and CSCR hospital Dataset.zip")
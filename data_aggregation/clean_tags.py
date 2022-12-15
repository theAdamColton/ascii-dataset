from glob import glob
from tqdm import tqdm

files = glob("**/*.txt", recursive=True)

for file in tqdm(files):
    txt: str = None
    with open(file, "r") as f:
        txt = f.read()

        # jgs is a prevalent artists in this dataset.
        # I feel bad, but I want this dataset to focus on 
        # the art.

        replacements = ["jgs", "JRS", "Veilleux", "Normand", "mrf", "/akg", "dp", "fsc", "hjw", "-Felix Lee-", "SSt", "AsH/HJ98", "sjw", "Pru", "ejm"]

        for s in replacements:
            txt = txt.replace(s, len(s) * " ")

    with open(file, "w") as f:
        f.write(txt)


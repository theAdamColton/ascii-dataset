"""
This reads each file into a set and checks for duplicates

For best effect, make sure all files have been padded and are uniform
"""

import os
import subprocess
import glob

textfiles = glob.glob("data/**/*.txt", recursive=True)

encountered = {}

for file in textfiles:
    with open(file, "r") as f:
        a = f.read()

    if a in encountered:
        print(file, encountered[a])
        subprocess.run(["rm", "-f", file])
    else:
        encountered[a] = file

import os
from pprint import pprint
import matplotlib.pyplot as plt

ROOT_DIR = "./logs"

RESULTS_DICT = {}

for root, dirs, files in os.walk(ROOT_DIR):
    for name in files:
        file_path = os.path.join(root, name)

        if file_path.endswith(".log"):
            with open(file_path, "r") as f:
                lines = f.readlines()
                last_line = lines[-1]
                print(last_line)
                if "wikitext" in last_line:
                    _, ppl = last_line.split("wikitext")
                    ppl = float(ppl)
                    RESULTS_DICT[name] = ppl
                else:
                    print(f"GOT OOM in {file_path}")
pprint(RESULTS_DICT)

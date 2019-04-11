"""This small script takes in Action classification
parameters  as depicted in the google-spreadsheet used
in the autograph project, and outputs it as Python data
that can be incorporated in the Autograph scripts


TODO: Read the spreadsheet data directly from within
the Autograph main script.

"""

import csv
from pathlib import Path
from pprint import pformat

filepath = Path("AUTOGRAPH TABELA INTENSIDADES - repertorio.csv")
headers = "name speed pressure direction size frames letter_notes new_file_name".split()
reader = csv.reader(filepath.open())
raw_data = list(reader)
data = [dict(zip(headers, row))  for row in raw_data[5:]]

per_letter_data = {}
for row in data:
    letter_notes = row.get("letter_notes", "*")
    letter = letter_notes.strip().lower()[0] if letter_notes.strip() else "*"
    if letter == "*" and row.get("name", "").count("_") >= 2:
        tmp = row["name"].split("_")[1]
        if len(tmp) == 1:
            letter = tmp.lower()
    per_letter_data.setdefault(letter, []).append(row)

# This generated file should be placed where Blender's Python can find it -
# (for example ~/.config/blender/2.79/scripts/modules/ )

Path("autograph_action_data.py").write_text("data = \\\n" + pformat(per_letter_data))

"""This small script takes in Action classification
parameters  as depicted in the google-spreadsheet used
in the autograph project, and outputs it as Python data
that can be incorporated in the Autograph scripts


TODO: Read the spreadsheet data directly from within
the Autograph main script.

"""

import csv
from pathlib import Path

# Filename bellow is as exported by the spreadsheet where each
# action sequence is paramterized by the artistic author.
# The file is exported as CSV from google-drive/sheets and read by this script

csv_path = "AUTOGRAPH TABELA INTENSIDADES - repertorio.csv"

def get_data_from_csv_file(csv_file):
    headers = "name speed pressure direction size old_frames letter_notes letter frames".split()

    raw_data = list(csv.reader(csv_file))
    # Skip spreadsheet header rows:
    data = [dict(zip(headers, row))  for row in raw_data[5:]]

    per_letter_data = {}
    for row in data:
        row.pop("old_frames", "")
        letter_notes = row.get("letter_notes", "*")
        # letter = letter_notes.strip().lower()[0] if letter_notes.strip() else "*"
        letter = row.get("letter").strip()
        if letter == "#":
            letter = " "

        #if letter == "*" and row.get("name", "").count("_") >= 2:
            #tmp = row["name"].split("_")[1]
            #if len(tmp) == 1:
                #letter = tmp.lower()
        if not row.get("pressure"):
            # Do not annotate letters not yet parametrized
            continue
        per_letter_data.setdefault(letter, []).append(row)
    return per_letter_data


def write_static_file(data, path="autograph_action_data.py"):
    # This generated file should be placed where Blender's Python can find it -
    # (for example ~/.config/blender/2.79/scripts/modules/ )

    from pprint import pformat

    Path(path).write_text("data = \\\n" + pformat(data))


if __name__ == "__main__":
    write_static_file(get_data_from_csv_file(Path(csv_path).open()))

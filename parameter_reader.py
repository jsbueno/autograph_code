"""This small script takes in Action classification
parameters  as depicted in the google-spreadsheet used
in the autograph project, and outputs it as Python data
that can be incorporated in the Autograph scripts


"""

import csv

import io

import sys
import requests
from pathlib import Path


# Filename bellow is as exported by the spreadsheet where each
# action sequence is paramterized by the artistic author.
# The file is exported as CSV from google-drive/sheets and read by this script

csv_path = "AUTOGRAPH TABELA INTENSIDADES - repertorio.csv"

def get_data_from_csv(csv_file):
    headers = "name speed pressure direction size old_frames letter_notes letter frames selected speed_factor".split()

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


def download_intensity_table(url):
    error = False

    try:
        content = requests.get(url, verify=False).text

    except Exception as err:
        print(err, file=sys.stderr)
        error = True
    if error:
        print(f"erro ao baixar planilha: {error}")
        raise RuntimeError("Could not get online data")
    zz = io.StringIO(content)
    return zz


def get_online_actions(url):
    csv_data = download_intensity_table(url)
    action_data = get_data_from_csv(csv_data)
    return action_data


if __name__ == "__main__":
    write_static_file(get_data_from_csv(Path(csv_path).open()))

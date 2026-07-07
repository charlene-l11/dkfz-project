# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import re
import subprocess
import sys
from pathlib import Path


def parse_entries(bib_content: str) -> dict[str, dict[str, str]]:
    """
    Parse the bibtex file content.

    Args:
        bib_content: The content of the bib file as a string.

    Returns: A dictionary of entries, where the key is the name of the entry (latex key) and the value is another dictionary with the content of the entry containing all the fields.
    """
    assert len(bib_content) > 0, "Empty BibTeX file"

    # We operate on the cleaned version from biber, so a simple regex does the job to parse the bib file
    entries = {}
    for entry_text in bib_content.split("@"):
        if entry_text == "":
            continue

        match = re.search("{([^,]+),", entry_text)
        assert match is not None, f"Could not find the name for the entry {entry_text}"
        name = match.group(1)

        entry = {}
        for key, value in re.findall(r"(\w+)\s*=\s*{(.+?)},\n", entry_text):
            entry[key] = value.replace("{", "").replace("}", "")

        assert name not in entries, f"Duplicate entry {name}"
        entries[name] = entry

    return entries


def check_entries(entries: dict[str, dict[str, str]]) -> None:
    """
    Check the bibtex entries for common issues. Found issues will raise assertion errors.

    Args:
        entries: The dictionary of entries as returned by `parse_entries()`.
    """
    errors = []

    dois = set()
    titles = set()
    for name, entry in entries.items():
        if "doi" in entry:
            if entry["doi"].startswith("http"):
                errors.append(
                    f"{name}: DOI {entry['doi']} starts with http (does are automatically links even without the http"
                    " prefix)"
                )
            if re.search(r"^\d", entry["doi"]) is None:
                errors.append(f"{name}: DOI {entry['doi']} does not start with a digit")
            if entry["doi"] in dois:
                errors.append(f"{name}: Duplicate DOI {entry['doi']}")
            dois.add(entry["doi"])

        if "title" in entry and "date" in entry:
            # title alone is not unique
            title_date = entry["title"] + entry["date"]
            if title_date in titles:
                errors.append(f"{name}: Duplicate title {entry['title']} and date {entry['date']}")
            titles.add(title_date)

        if "url" in entry:
            if entry["url"].endswith("/"):
                errors.append(
                    f"{name}: URL {entry['url']} ends with / (this is unnecessary but will appear in the bibliography"
                    " in the PDF)"
                )
            if "doi" in entry["url"] and "doi" in entry:
                errors.append(
                    f"{name}: URL {entry['url']} contains a doi link but a doi field is also present (this is"
                    " unnecessary as the doi already contains the link)"
                )

    assert len(dois) > 0, "No DOIs found"
    assert len(titles) > 0, "No titles found"
    assert len(errors) == 0, "\n".join(errors)


if __name__ == "__main__":
    returncode = 0

    for file in sys.argv[1:]:
        print(f"Running biber tool on {file}")

        # See biber manual for options: https://ctan.ebinger.cc/tex-archive/biblio/biber/base/documentation/biber.pdf#page=45
        res = subprocess.run(
            "biber --tool --output-align --output-indent=2 --output-fieldcase=lower --isbn-normalise --output-file"
            f" {file} {file}",
            shell=True,
            text=True,
            capture_output=True,
        )

        # Make biber tool work with end-of-file-fixer hook
        file = Path(file)
        file.write_text(file.read_text().removesuffix("\n"))

        file_tmp = file.with_suffix(".bib.blg")
        if file_tmp.exists():
            file_tmp.unlink()

        if res.stderr != "":
            print(res.stderr)
            returncode = 1
        if res.stdout != "":
            print(res.stdout)

        if res.returncode != 0:
            returncode = 2

        if "WARN" in res.stdout:
            returncode = 3

    if returncode == 0:
        try:
            print("Running checks on bib files")
            bib_content = "\n".join([Path(file).read_text() for file in sys.argv[1:]])
            entries = parse_entries(bib_content)
            check_entries(entries)
        except Exception as e:
            print("ERROR: found issues in the bib files:")
            print(e)
            returncode = 4

    sys.exit(returncode)

# Instructions for setting up a dataset of ASCII art for training:
* Scrape art using the multiple scripts, `scrape*.py`
* Filter out files that are too large or small using `filter_data.py`
* Filter out non-acceptable ASCII characters using `filter_ascii_chars.py`
* Pad every piece of art with `pad_data.py` to make each line in each piece have the same number of characters as it's longest line.
* Optionally, additional filtering can be done with `clean_tags.py`, which removes some artists attribution signature, (sorry!).
* Optionally, artwork can be organized in a directory format that represents the labels for each piece.
* `remove_duplicates.py` removes all duplicates, which can be quite many.

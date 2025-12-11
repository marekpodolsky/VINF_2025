import os
import re
import csv
import logging
import html

# Nastavenie logovania
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

INPUT_DIR = "pages"
OUTPUT_FILE = "nfl_players_extracted.csv"
BATCH_SIZE = 10

DOTALL = re.DOTALL
IGNORECASE = re.IGNORECASE

COMPILED_REGEX = {
    # Meno
    "NAME_SPAN": re.compile(r'<h1[^>]*>.*?<span>(.*?)</span>', DOTALL),
    "NAME_H1": re.compile(r'<h1[^>]*>(.*?)</h1>', DOTALL),

    # Meta sekcia
    "META_SECTION": re.compile(r'<div[^>]+id=["\']meta["\'][^>]*>(.*?)<script', DOTALL),
    "META_DIVS": re.compile(r'<div[^>]+>(.*?)</div>', DOTALL),

    # Detaily
    "POSITION": re.compile(r'<strong>Position(?:s)?</strong>:\s*([A-Za-z/,\- ]+)'),
    "HEIGHT": re.compile(r'<span>(\d{1,2}-\d{1,2})</span>'),
    "WEIGHT": re.compile(r'<span>(\d{2,3})lb</span>'),
    "BORN_YEAR": re.compile(r'<strong>Born:</strong>.*?(\d{4})', DOTALL),

    # Miesto narodenia
    "BORN_PLACE_US": re.compile(r'in&nbsp;([^,]+),&nbsp;<a [^>]+>([A-Z]{2})</a>'),
    "BORN_PLACE_INTL": re.compile(r'in&nbsp;([^,]+),&nbsp;<a [^>]+>([^<]+)</a>'),

    # Školy
    "COLLEGE": re.compile(r'<strong>College</strong>:\s*<a[^>]+>([^<]+)</a>'),
    "HIGH_SCHOOL": re.compile(r'<strong>High School</strong>:\s*<a[^>]+>([^<]+)</a>'),

    # Draft
    "DRAFT": re.compile(
        r'<strong>Draft</strong>:.*?<a[^>]*?>([^<]+)</a>.*?/years/(\d{4})/draft\.htm',
        DOTALL
    ),

    # Smrť/Tím
    "DIED": re.compile(r'<strong>Died:</strong>.*?(\d{4})', DOTALL),
    "TEAM": re.compile(
        r'<strong>Team(?:s)?</strong>:\s*(?:<span>)?<a[^>]+>([^<]+)</a>',
        DOTALL | IGNORECASE
    ),
}

# Funkcia extrakcie
def extract_player_data(html_content):
    R = COMPILED_REGEX

    # Meno
    name_match = R["NAME_SPAN"].search(html_content)
    if not name_match:
        name_match = R["NAME_H1"].search(html_content)

    name = re.sub(r"<.*?>", "", name_match.group(1)).strip() if name_match else "Unknown"

    # Meta sekcia
    meta_html = ""
    meta_match = R["META_SECTION"].search(html_content)
    if meta_match:
        meta_html = meta_match.group(1)
    else:
        for section in R["META_DIVS"].findall(html_content):
            if 'id="meta"' in section:
                meta_html += section

    if not meta_html:
        logging.warning(f"No meta section found for {name}")
        return {
            "Player": name, "Position": "N/A", "Height": "N/A", "Weight": "N/A",
            "BornYear": "N/A", "BornCity": "N/A", "BornState": "N/A",
            "CurrentTeam": "Unknown Status (Meta Missing)",
            "DiedYear": "N/A",
            "College": "N/A", "HighSchool": "N/A",
            "DraftTeam": "Undrafted", "DraftYear": "Undrafted",
        }

    # Extrakcia detailov
    position = R["POSITION"].search(meta_html)
    position = position.group(1).strip() if position else "N/A"

    height_match = R["HEIGHT"].search(meta_html)
    height = height_match.group(1) if height_match else "N/A"

    weight_match = R["WEIGHT"].search(meta_html)
    weight = weight_match.group(1) + "lb" if weight_match else "N/A"

    born_match = R["BORN_YEAR"].search(meta_html)
    born_year = born_match.group(1) if born_match else "N/A"

    born_city = "N/A"
    born_state = "N/A"
    born_place_match_us = R["BORN_PLACE_US"].search(meta_html)
    if born_place_match_us:
        born_city = born_place_match_us.group(1).strip()
        born_state = born_place_match_us.group(2).strip()
    else:
        born_place_match_intl = R["BORN_PLACE_INTL"].search(meta_html)
        if born_place_match_intl:
            born_city = born_place_match_intl.group(1).strip()
            born_state = born_place_match_intl.group(2).strip()

    college_match = R["COLLEGE"].search(meta_html)
    college = college_match.group(1).strip() if college_match else "N/A"

    hs_match = R["HIGH_SCHOOL"].search(meta_html)
    highschool = hs_match.group(1).strip() if hs_match else "N/A"

    draft_match = R["DRAFT"].search(meta_html)
    draft_team = draft_match.group(1).strip() if draft_match else "Undrafted"
    draft_year = draft_match.group(2).strip() if draft_match else "Undrafted"

    current_team = "Retired"
    died_year_result = "Alive"

    died_match = R["DIED"].search(meta_html)
    if died_match:
        current_team = "Died"
        died_year_result = died_match.group(1).strip()
    else:
        team_match = R["TEAM"].search(meta_html)
        if team_match:
            current_team = team_match.group(1).strip()

    result = {
        "Player": name, "Position": position, "Height": height, "Weight": weight,
        "BornYear": born_year, "BornCity": born_city, "BornState": born_state,
        "CurrentTeam": current_team, "DiedYear": died_year_result,
        "College": college, "HighSchool": highschool,
        "DraftTeam": draft_team, "DraftYear": draft_year,
    }

    # Dekódovanie HTML entít
    for key, value in result.items():
        if isinstance(value, str):
            result[key] = html.unescape(value)

    return result

# Funkcia pre extrakciu čísel pre numerické radenie
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

# Hlavná funkcia
def process_all_pages():
    if not os.path.exists(INPUT_DIR):
        logging.error(f"Directory '{INPUT_DIR}' not found. Please create it and place HTML files inside.")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".html")]
    if not files:
        logging.warning("No HTML files found in input directory.")
        return

    # Súbory triedim, aby som ich spracoval v správnom poradí (napr. A_1.html, A_2.html… namiesto A_1.html, A_10.html…)
    sorted_files = sorted(files, key=natural_sort_key)

    files_to_process = sorted_files
    mode = 'w'

    logging.info("Starting a **NEW** extraction run. Output file will be overwritten.")

    # Získanie Fieldnames z prvej platnej extrakcie
    fieldnames = None
    if not files_to_process:
        logging.error("No files available to process after sorting.")
        return

    try:
        # Extrahujem prvý súbor len kvôli získaniu kľúčov
        with open(os.path.join(INPUT_DIR, files_to_process[0]), "r", encoding="utf-8") as f:
            first_record = extract_player_data(f.read())
        fieldnames = first_record.keys()
    except Exception as e:
        logging.warning(f"Failed to extract from the first file to get headers. Using default keys. Error: {e}")
        # Ak extrakcia zlyhá
        example_record = extract_player_data("")
        fieldnames = example_record.keys()

    # Spracovanie a priebežné ukladanie
    total_processed_count = 0
    total_files = len(sorted_files)

    with open(OUTPUT_FILE, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Pri režime 'w' vždy zapíšem hlavičku
        writer.writeheader()
        logging.info(f"Created/Overwritten file '{OUTPUT_FILE}' with headers.")

        data_batch = []

        # i reprezentuje 1-based index (1, 2, 3...)
        for i, file in enumerate(files_to_process, start=1):
            path = os.path.join(INPUT_DIR, file)
            try:
                with open(path, "r", encoding="utf-8") as file_handle:
                    html_content = file_handle.read()

                player_data = extract_player_data(html_content)
                data_batch.append(player_data)

                total_processed_count += 1

                logging.info(f"[{i}/{total_files}] Extracted: {player_data['Player']} from {file}")

                # Ukladanie v dávkach
                if len(data_batch) >= BATCH_SIZE:
                    writer.writerows(data_batch)
                    f.flush()  # Vynúti zápis na disk
                    data_batch = []
                    logging.info(f"--- Batch saved. Total extracted: {total_processed_count} ---")

            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

        # Uloženie zvyšných záznamov po dokončení cyklu
        if data_batch:
            writer.writerows(data_batch)
            f.flush()
            logging.info(f"--- Final batch saved. ---")

    logging.info(f"Extraction finished. Total players extracted: {total_processed_count}")


if __name__ == "__main__":
    logging.info("=== NFL Extractor Started (Always NEW Run) ===")
    process_all_pages()
    logging.info("=== NFL Extractor Finished ===")
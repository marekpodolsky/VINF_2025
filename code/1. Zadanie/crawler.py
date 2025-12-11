import os
import time
import random
import logging
import re
import string
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Konfigurácia logovania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
BASE_URL = "https://www.pro-football-reference.com/players/"
OUTPUT_DIR = "pages"

# Inicializácia WebDriveru
def init_driver():
    chrome_options = Options()
    chrome_options.page_load_strategy = 'normal'
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=chrome_options)

# Extrakcia odkazov hráčov z list stránky
def get_player_links(driver, letter):
    url = BASE_URL + letter + "/"
    logging.info(f"Fetching list page: {url}")
    driver.get(url)
    time.sleep(random.uniform(3.0, 3.5))

    html = driver.page_source

    # Zahrnutie komentovaných sekcií
    commented_sections = re.findall(r"", html, re.DOTALL)
    for section in commented_sections:
        if 'data-append-csv' in section:
            html += section

    # Regex na odkazy
    pattern = r'<a href="(/players/[A-Z]/[^"]+\.htm)">[^<]+</a>'
    links = re.findall(pattern, html)

    # Odstránenie duplicity a doplnenie plných URL
    full_links = list(dict.fromkeys(["https://www.pro-football-reference.com" + l for l in links]))

    logging.info(f"Found {len(full_links)} player links for letter {letter}")
    return full_links


# Uloženie HTML súboru do priečinka
def save_page(html, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logging.info(f"Saved {filename}")


# Crawl hráčov pre dané písmeno
def crawl_players_by_letter(letter):
    driver = init_driver()
    try:
        player_links = get_player_links(driver, letter)
        if not player_links:
            logging.warning(f"No players found for letter {letter}. Skipping.")
            return

        len_players = len(player_links)

        logging.info(
                f"*** STARTING: Crawling letter '{letter}' with {len_players} players ***")

        # Iterácia prechádza cez všetky odkazy od prvého
        for idx_offset, link in enumerate(player_links):
            # idx je číslo, ktoré sa použije pre názov súboru (1-based index)
            idx = idx_offset + 1

            logging.info(f"[{letter}] Fetching ({idx}/{len_players}): {link}")

            # Vykonanie požiadavky
            driver.get(link)

            # Okamžité uloženie HTML
            html = driver.page_source
            filename = f"{letter}_{idx}.html"
            save_page(html, filename)  # Súbor sa uloží okamžite po crawle

            # Pauza
            time.sleep(random.uniform(3, 3.5))

        logging.info(f"Finished crawling letter {letter}, total {len_players} pages saved.")

    finally:
        driver.quit()


if __name__ == "__main__":
    # Vždy prechádzam postupne cez všetky písmená od 'A' po 'Z'
    all_letters = string.ascii_uppercase
    letters_to_crawl = all_letters

    logging.info("=== NFL Crawler Started (Starting from Letter A, Player 1) ===")

    for i, letter in enumerate(letters_to_crawl):
        # Pridanie ďalšej pauzy medzi každým písmenom (okrem úplne prvého v tomto behu)
        if i != 0:
            logging.info("--- Long cooldown pause between letters ---")
            time.sleep(random.uniform(3.0, 3.5))

        crawl_players_by_letter(letter)

    logging.info("===================================")
    logging.info("ALL CRAWLING TASKS COMPLETED")
    logging.info("===================================")
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.sql.functions import col, udf, when, lit
import os
import re

# Nastavenie cesty k pythonu (pre Spark)
os.environ['PYSPARK_PYTHON'] = 'python'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'

# Inicializácia Sparku
spark = SparkSession.builder \
    .appName("NFLPlayersExtractor") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.files.maxPartitionBytes", "67108864") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.default.parallelism", "50") \
    .getOrCreate()

# Definovanie cesty k súborom
wiki_path = "enwiki-latest-pages-articles.xml.bz2"
output_path = "nfl_players_FINAL.csv"

# Funkcie na čistenie dát
def clean_player_name(player_name):
    # Vyčistenie mena hráča a odstránenie textu v zátvorkách
    if not player_name:
        return ""
    clean_name = re.sub(r'\s*\([^)]*\)', '', player_name)
    return clean_name.strip()

def extract_surname(player_name):
    # Extrakcia priezviska hráča (posledné slovo v mene)
    if not player_name:
        return ""
    clean_name = clean_player_name(player_name)
    parts = clean_name.strip().split()
    return parts[-1] if parts else ""

def has_nfl_categories(page_text):
    # Kontrola, či text stránky obsahuje kategórie súvisiace s hráčmi NFL
    nfl_category_patterns = [
        r"\[\[Category:.*NFL players\]\]",
        r"\[\[Category:.*National Football League players\]\]",
        r"\[\[Category:.*(American|Canadian) football.*players\]\]",
    ]

    # Zoznam všetkých tímov NFL pre špecifickú kontrolu kategórií
    nfl_teams = [
        "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
        "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
        "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
        "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
        "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
        "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
        "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
        "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders"
    ]

    for team in nfl_teams:
        if f"[[Category:{team}" in page_text:
            return True

    for pattern in nfl_category_patterns:
        if re.search(pattern, page_text):
            return True

    return False

def clean_wiki_markup(text):
    # Čistenie textu Wiki od značiek a HTML entít.
    if not text:
        return ""

    text = re.sub(r"&lt;ref.*?&lt;/ref&gt;", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref.*?>.*?</ref>", "", text, flags=re.DOTALL)

    text = re.sub(r"<ref[^>/]*/>", "", text)

    text = re.sub(r"\bref\s*\{\{.*?\}\}", "", text, flags=re.DOTALL | re.IGNORECASE)

    text = re.sub(r"\[\d+\]", "", text)

    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)

    text = re.sub(r"\[https?://[^\s]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", "", text)

    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    text = re.sub(r"'{2,}", "", text)

    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&[a-z]+;", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# Extrakcia prvého paragrafu wiki popisu hráča
def extract_first_paragraph(page_text):
    if not page_text:
        return ""

    try:
        sentence_match = re.search(r"'''.+?'''.+?(\.\s|\!|\?)", page_text, re.DOTALL)
        if sentence_match:
            first_sentence = sentence_match.group(0)
            cleaned_sentence = clean_wiki_markup(first_sentence)
            return re.sub(r'\s+', ' ', cleaned_sentence).strip()
        return ""
    except Exception as e:
        print(f"Error extracting first paragraph: {e}")
        return ""

# Extrakcia hodnoty Infoboxu
def extract_infobox_field(page_text, field_name):
    if not page_text:
        return ""
    pattern = rf"\|\s*{re.escape(field_name)}\s*=\s*(.*?)(?=(?:\n\s*\||\s*\|\s*\w+\s*=)|\n\s*\}}|\Z)"
    match = re.search(pattern, page_text, re.DOTALL | re.IGNORECASE)
    if match:
        raw_value = match.group(1).strip()
        if re.search(r"\|\s*\w+\s*=", raw_value):
            raw_value = ""
        raw_value = re.sub(r"\s*\|\s*$", "", raw_value)
        return clean_wiki_markup(raw_value)
    return ""

# Extrakcia váhy hráča (rôzne typy)
def extract_weight(page_text):
    weight_fields = ["weight_lb", "weight_lbs", "weight", "weight_kg"]
    for f in weight_fields:
        value = extract_infobox_field(page_text, f)
        if value and value.strip():
            m = re.match(r"^(\d{2,3})\s*$", value)
            if m and f in ["weight_lb", "weight_lbs", "weight"]:
                return f"{m.group(1)} lb"
            return value
    return ""

# Extrakcia roku narodenia
def extract_birth_year(text):
    birth_patterns = [
        (r"\{\{Birth date and age\|(\d{4})\|(\d{1,2})\|(\d{1,2})\}\}"),
        (r"\{\{Birth date and age\|[^}]*?\|(\d{4})\|(\d{1,2})\|(\d{1,2})\}\}"),
        (r"\{\{Birth-date\|(\d{4})\|(\d{1,2})\|(\d{1,2})\}\}"),
        (r"\{\{bda\|(\d{4})\|(\d{1,2})\|(\d{1,2})\}\}"),
        (r"\{\{birth date\|(\d{4})\|(\d{1,2})\|(\d{1,2})\|.*?\}\}"),
        (r"\{\{Birth date\|(\d{4})\|(\d{1,2})\|(\d{1,2})\}\}"),
        (r"birth_date\s*=\s*(\d{4})-(\d{1,2})-(\d{1,2})"),
        (r"birth_date\s*=\s*\{\{birth date and age\|(\d{4})\|(\d{1,2})\|(\d{1,2})\}\}"),
        (r"\{\{Birth year and age\|(\d{4})\}\}"),
        (r"\{\{born in\|(\d{4})\}\}"),
        (r"\{\{birth year and age\|(\d{4})\}\}"),
        (r"born\s*=\s*\{\{birth year and age\|(\d{4})\}\}"),
        (r"\{\{Birth\|\s*year\s*=\s*(\d{4})\s*\|\s*month\s*=\s*(\d{1,2})\s*\|\s*day\s*=\s*(\d{1,2})\s*\}\}"),
        (r"\{\{death date and age\|\d{4}\|\d{1,2}\|\d{1,2}\|(\d{4})\|\d{1,2}\|\d{1,2}\}\}"),
    ]
    for pattern in birth_patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if len(groups) >= 1:
                return groups[0]
    return None

# Hlavná extrakčná funkcia
def extract_nfl_players_robust():
    print("Starting robust NFL players extraction...")
    rdd = spark.sparkContext.textFile(wiki_path, minPartitions=100)
    print(f"Total partitions: {rdd.getNumPartitions()}")

    def process_page(lines):
        page_text = "\n".join(lines)

        strong_nfl_indicators = [
            "Infobox NFL player", "Infobox NFL biography",
            "Infobox American football player", "{{NFL player",
            "{{American football name",
        ]

        non_nfl_indicators = [
            "Infobox football biography", "Infobox footballer", "Infobox soccer",
            "Infobox Australian rules football", "Infobox Gaelic football",
            "Infobox baseball player", "Infobox baseball biography",
            "Infobox NBA player", "Infobox basketball player",
        ]

        has_strong_nfl_indicator = any(ind in page_text for ind in strong_nfl_indicators)
        has_nfl_categories_flag = has_nfl_categories(page_text)
        has_non_nfl_indicator = any(ind in page_text for ind in non_nfl_indicators)

        if not has_strong_nfl_indicator:
            if not has_nfl_categories_flag or has_non_nfl_indicator:
                return []
        if has_non_nfl_indicator and not has_strong_nfl_indicator:
            return []

        title_match = re.search(r"<title>(.*?)</title>", page_text)
        if not title_match:
            return []

        title = title_match.group(1)
        if any(x in title for x in ["Category:", "Template:", "Wikipedia:", "User:", "File:", "MediaWiki:"]):
            return []

        birth_year = extract_birth_year(page_text)
        info_paragraph = extract_first_paragraph(page_text)
        birth_place = extract_infobox_field(page_text, "birth_place")
        height_ft = extract_infobox_field(page_text, "height_ft")
        height_in = extract_infobox_field(page_text, "height_in")
        weight_lb = extract_weight(page_text)
        college = extract_infobox_field(page_text, "college")
        high_school = extract_infobox_field(page_text, "high_school")
        draft_year = extract_infobox_field(page_text, "draftyear")
        draft_round = extract_infobox_field(page_text, "draftround")

        if birth_year:
            clean_name = clean_player_name(title)
            return [(clean_name, birth_year, birth_place,
                     height_ft, height_in, weight_lb, college, high_school,
                     draft_year, draft_round, info_paragraph)]
        return []

# Zoskupenie riadkov do kompletných Wiki stránok
    def group_pages(iterator):
        current_page = []
        in_page = False
        page_start = re.compile(r"<page>")
        page_end = re.compile(r"</page>")

        for line in iterator:
            if page_start.search(line):
                in_page = True
                current_page = [line]
            elif page_end.search(line):
                if in_page:
                    current_page.append(line)
                    in_page = False
                    yield current_page
                    current_page = []
            elif in_page:
                current_page.append(line)
        if current_page:
            yield current_page

    print("Processing pages and extracting player information...")
    player_data = rdd.mapPartitions(group_pages) \
        .flatMap(process_page) \
        .distinct()

    schema = StructType([
        StructField("player_name", StringType(), True),
        StructField("birth_year", StringType(), True),
        StructField("birth_place", StringType(), True),
        StructField("height_ft", StringType(), True),
        StructField("height_in", StringType(), True),
        StructField("weight_lb", StringType(), True),
        StructField("college", StringType(), True),
        StructField("high_school", StringType(), True),
        StructField("draft_year", StringType(), True),
        StructField("draft_round", StringType(), True),
        StructField("info", StringType(), True)
    ])

    players_df = spark.createDataFrame(player_data, schema)

    nullable_cols = [c for c in players_df.columns if c not in ["player_name", "birth_year"]]
    for col_name in nullable_cols:
        players_df = players_df.withColumn(
            col_name,
            when((col(col_name) == "") | col(col_name).isNull(), lit("N/A")).otherwise(col(col_name))
        )

    extract_surname_udf = udf(extract_surname, StringType())
    players_df = players_df.withColumn("surname", extract_surname_udf("player_name"))
    players_df = players_df.orderBy("surname")
    return players_df.drop("surname")

if __name__ == "__main__":
    try:
        print("=" * 70)
        print("NFL Players Extractor (Final Revised: Improved birth_place & weight parsing + cleaned citations)")
        print("=" * 70)

        print("\nMethod 1: Robust RDD Approach")
        players_df = extract_nfl_players_robust()
        player_count = players_df.count()
        print(f"Found {player_count} NFL players with complete data")

        # Zníženie partícií na 1 a zapísanie výsledkov ako CSV
        if player_count > 0:
            output_path_final = output_path
            players_df.coalesce(1) \
                .write \
                .mode("overwrite") \
                .option("header", "true") \
                .csv(output_path_final)
            print(f"\nResults saved to: {output_path_final} (sorted by surname A-Z)")
        else:
            print("No players found with robust approach.")

    except Exception as e:
        print(f"Main extraction failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        spark.stop()
        print("\nProcessing completed!")

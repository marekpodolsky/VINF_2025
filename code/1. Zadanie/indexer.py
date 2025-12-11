import csv
import json
import os
import re
import sys
from collections import defaultdict
from math import log, sqrt

# Konfigurácia a súbory
CSV_SOURCE = "nfl_players_extracted.csv"
OUTPUT_DIR = "indexer_data"

FILE_INDEX_POSTINGS = "inverted_indexes.json"
FILE_METADATA = "docs.json"
FILE_VOCAB = "vocabulary.json"
FILE_DF = "df.json"
FILE_TF = "tf.json"
FILE_TFIDF_WEIGHTS = "tfidf_standard_probabilistic.json"
FILE_NORMS = "doc_norms_standard_probabilistic.json"

# Funkcie pre indexovanie
def process_text_to_tokens(text):
    # Malé písmená a extrakcia alfanumerických tokenov (neaplikuje sa žiadne filtrovanie podľa dĺžky)
    if not text:
        return []
    text = text.lower()
    # Získanie alfanumerických reťazcov vrátane pomlčiek a apostrofov
    tokens = re.findall(r"[a-z0-9]+(?:['\-][a-z0-9]+)*", text)
    return tokens

def load_data_from_csv(path_to_csv):
    # Načítanie dát z CSV a ich príprava pre indexovanie a metadáta.
    player_documents = {}  # ID -> text pre index
    player_meta = {}  # ID -> metadáta

    # Definícia polí pre text indexu (všetky columns z CSV)
    INDEX_FIELDS = [
        "Player", "Position", "Height", "Weight", "BornYear",
        "BornCity", "BornState", "CurrentTeam", "DiedYear",
        "College", "HighSchool", "DraftTeam", "DraftYear"
    ]

    try:
        with open(path_to_csv, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for doc_counter, record in enumerate(reader, start=1):

                # Kontrola na vylúčenie nekorektných záznamov (pokiaľ je záznam corrupted)
                player_name = record.get("Player", "")
                current_team = record.get("CurrentTeam", "")

                # Vylúčenie záznamov, ktoré sú "Unknown" alebo "Missing"
                if player_name == "Unknown" or current_team == "Unknown Status (Meta Missing)":
                    print(
                        f"  [INFO] Vylúčenie riadku #{doc_counter} (Player: '{player_name}', Team: '{current_team}') z indexovania - dáta chýbajú/sú neznáme.")
                    continue  # Preskočí celý cyklus pre tento nekorektný riadok

                doc_id = str(doc_counter)

                # Vytvorenie indexovateľného textu
                doc_parts = [record.get(key, "") for key in INDEX_FIELDS]
                index_text = " ".join([p for p in doc_parts if p])

                # Uloženie dát (metadáta)
                player_documents[doc_id] = index_text
                player_meta[doc_id] = {
                    "meno": record.get("Player", ""),
                    "pozicia": record.get("Position", ""),
                    "tim": record.get("CurrentTeam", ""),
                    "rok_draftu": record.get("DraftYear", ""),
                    "vsetky_data": record
                }
        return player_documents, player_meta
    except FileNotFoundError:
        print(f"--- CHYBA --- Vstupný CSV súbor '{path_to_csv}' nebol nájdený.")
        sys.exit(1)
    except Exception as e:
        print(f"--- CHYBA --- Pri čítaní CSV: {e}.")
        sys.exit(1)

def create_index_components(documents):
    # Vytvorí Inverted Index, TF (Term Frequency) a DF (Document Frequency).
    index_postings = defaultdict(set)
    df_counts = defaultdict(int)
    tf_counts = defaultdict(lambda: defaultdict(int))

    for doc_id, text in documents.items():
        tokens = process_text_to_tokens(text)

        # Počítanie TF
        for term in tokens:
            tf_counts[doc_id][term] += 1

        # Počítanie DF a Postings List
        for term in set(tokens):  # Iba unikátne termy pre DF
            df_counts[term] += 1
            index_postings[term].add(doc_id)

    # Konverzia setov v Postings Liste na zotriedené zoznamy
    final_index = {t: sorted(list(postings), key=lambda x: int(x)) for t, postings in index_postings.items()}

    # Konverzia na štandardné dict pre JSON serializáciu
    tf_data = {doc_id: dict(term_counts) for doc_id, term_counts in tf_counts.items()}
    df_data = dict(df_counts)

    return final_index, tf_data, df_data

# IDF a TF-IDF váhy
def get_idf_standard(df_t, total_docs):
    # Standard IDF: log(N / (df_t + 1))
    if df_t == 0: return 0.0
    return log(total_docs / (df_t + 1))

def get_idf_probabilistic(df_t, total_docs):
    # Probabilistic IDF: max(0, log((N - df_t) / df_t))
    if df_t == 0 or df_t == total_docs: return 0.0
    # Formula pre probabilistické váženie (log ratio)
    probabilistic_idf = log((total_docs - df_t) / df_t)
    return max(0.0, probabilistic_idf)

def calculate_tfidf_weights(tf_data, df_data, total_docs):
    # Výpočet TF-IDF váh pre obe IDF metódy.
    weights_standard = {}
    weights_probabilistic = {}

    for doc_id, term_freqs in tf_data.items():
        doc_vec_std = {}
        doc_vec_prob = {}

        for term, freq in term_freqs.items():
            df_val = df_data.get(term, 0)

            # Logaritmická TF zložka: 1 + log(freq)
            tf_log_weight = 1 + log(freq) if freq > 0 else 0.0

            # Váženie Standard IDF
            idf_std = get_idf_standard(df_val, total_docs)
            doc_vec_std[term] = tf_log_weight * idf_std

            # Váženie Probabilistic IDF
            idf_prob = get_idf_probabilistic(df_val, total_docs)
            doc_vec_prob[term] = tf_log_weight * idf_prob

        weights_standard[doc_id] = doc_vec_std
        weights_probabilistic[doc_id] = doc_vec_prob

    return {
        "standard": weights_standard,  # Použitie pôvodného kľúča "standard"
        "probabilistic": weights_probabilistic  # Použitie pôvodného kľúča "probabilistic"
    }

def compute_vector_norms(weights_dict):
    # Vypočíta euklidovské normy (dĺžky) vektorov pre obe metódy.
    norms_output = {}
    for method_name, weights in weights_dict.items():
        norms = {}
        for doc_id, vector in weights.items():
            squared_sum = sum(weight * weight for weight in vector.values())
            norms[doc_id] = sqrt(squared_sum) if squared_sum > 0 else 0.0
        norms_output[method_name] = norms
    return norms_output

# Ukladanie (JSON)
def custom_save_json(data_obj, file_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_path = os.path.join(OUTPUT_DIR, file_path)

    with open(full_path, "w", encoding="utf-8") as fh:
        # Minimalistický formát pre váhy (TF, TF-IDF)
        if file_path in [FILE_TFIDF_WEIGHTS, FILE_TF]:
            json.dump(data_obj, fh, ensure_ascii=False, separators=(',', ':'))
        # Čitateľný formát pre metadáta a index
        else:
            json.dump(data_obj, fh, ensure_ascii=False, indent=2)

# Hlavná logika
def run_indexing_process():
    # Úprava konzolového výstupu
    print("\n" + "=" * 55)
    print(f"  > SPÚŠŤAM INDEXER NFL HRÁČOV (Zdroj: {CSV_SOURCE})")
    print("=" * 55)

    # Načítanie a spracovanie dát
    documents, metadata = load_data_from_csv(CSV_SOURCE)
    total_documents = len(documents)

    if total_documents == 0:
        print("  [INFO] Indexovanie nebolo spustené (0 dokumentov).")
        print("=" * 55)
        return

    print(f"  [+] Načítané dokumenty: {total_documents}")

    # Vytvorenie základných komponentov indexu (Postings, TF, DF)
    print("\n  [PROCES 1/4] Budovanie indexu a frekvencií...")
    inv_index, tf_dict, df_dict = create_index_components(documents)
    vocabulary = sorted(inv_index.keys())

    print(f"  [>] Veľkosť slovníka (vocab): {len(vocabulary)} tokenov")

    # Výpočet TF-IDF váh
    print("\n  [PROCES 2/4] Výpočet váh TF-IDF (Standard a Probabilistic IDF)...")
    tfidf_weights = calculate_tfidf_weights(tf_dict, df_dict, total_documents)

    # Výpočet dĺžok vektorov (noriem)
    print("\n  [PROCES 3/4] Výpočet noriem vektorov pre ranking...")
    doc_norms = compute_vector_norms(tfidf_weights)

    # Ukladanie do JSON
    print("\n  [PROCES 4/4] Ukladanie indexových súborov do JSON...")

    custom_save_json(inv_index, FILE_INDEX_POSTINGS)
    custom_save_json(metadata, FILE_METADATA)
    custom_save_json(vocabulary, FILE_VOCAB)
    custom_save_json(df_dict, FILE_DF)
    custom_save_json(tf_dict, FILE_TF)
    custom_save_json(tfidf_weights, FILE_TFIDF_WEIGHTS)
    custom_save_json(doc_norms, FILE_NORMS)

    print("\n" + "-" * 55)
    print(f"  ÚSPECH: Index bol vytvorený v adresári '{OUTPUT_DIR}'")
    print("-" * 55)
    print("  Vytvorené súbory:")
    print(f"    - {FILE_INDEX_POSTINGS} (Invertovaný index)")
    print(f"    - {FILE_TFIDF_WEIGHTS} (TF-IDF váhy pre 2 metódy)")
    print(f"    - {FILE_NORMS} (Normy dokumentových vektorov)")
    print(f"    - {FILE_METADATA} (Metadáta dokumentov)")
    print(f"    - {FILE_VOCAB} (Slovník tokenov)")
    print("=" * 55)

if __name__ == "__main__":
    run_indexing_process()
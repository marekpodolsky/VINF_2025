import csv
import json
import os
import math
import re
import sys
from collections import defaultdict

# Výstupný adresár indexera
INDEX_DIR = "indexer_data"

POSTINGS_FILE = os.path.join(INDEX_DIR, "inverted_indexes.json")
TFIDF_FILE = os.path.join(INDEX_DIR, "tfidf_standard_probabilistic.json")
NORM_FILE = os.path.join(INDEX_DIR, "doc_norms_standard_probabilistic.json")
DF_FILE = os.path.join(INDEX_DIR, "df.json")
META_FILE = os.path.join(INDEX_DIR, "docs.json")
DATA_FILE = "nfl_players_extracted.csv"

# Tokenizácia
def tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-z0-9]+(?:['\-][a-z0-9]+)*", text)

# Logaritmické váženie TF: 1 + log(tf)
def tf_weight(tf):
    return 1 + math.log(tf) if tf > 0 else 0.0

# IDF Váženie (Standard)
def get_idf_standard(df_t, total_docs):
    if df_t == 0: return 0.0
    return math.log(total_docs / (df_t + 1))

# IDF Váženie (Probabilistic)
def get_idf_probabilistic(df_t, total_docs):
    if df_t == 0 or df_t == total_docs: return 0.0
    probabilistic_idf = math.log((total_docs - df_t) / df_t)
    return max(0.0, probabilistic_idf)

# Inverzia TF-IDF z Document:Term:Weight na Term:Document:Weight
def invert_tfidf_for_search(tfidf_data):
    inverted_index = defaultdict(lambda: defaultdict(dict))

    for method, doc_vecs in tfidf_data.items():
        for doc_id, term_weights in doc_vecs.items():
            for term, weight in term_weights.items():
                inverted_index[method][term][doc_id] = weight

    return dict(inverted_index)

# Načítanie a príprava všetkých indexových súborov a CSV dát
def load_data():
    try:
        with open(POSTINGS_FILE, "r", encoding="utf-8") as f:
            postings_index = json.load(f)
        with open(TFIDF_FILE, "r", encoding="utf-8") as f:
            tfidf_data = json.load(f)
        with open(NORM_FILE, "r", encoding="utf-8") as f:
            norms = json.load(f)
        with open(DF_FILE, "r", encoding="utf-8") as f:
            df_counts = json.load(f)
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        # Načítanie CSV dát
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data_list = list(csv.DictReader(f))
            data = {str(i + 1): row for i, row in enumerate(data_list)}

            # Invertovanie TF-IDF váh pre efektívny ranked_search
        inverted_tfidf_index = invert_tfidf_for_search(tfidf_data)

        print(f"Dáta a index úspešne načítané. Celkový počet dokumentov: {len(meta_data)}")

        return (
            postings_index,  # Inverted Index (term -> [doc_ids]) pre Boolean
            inverted_tfidf_index,  # Inverted TF-IDF (method -> term -> {doc_id: weight}) pre Ranked
            norms,  # Normy
            meta_data,  # Metadáta
            df_counts,  # Document Frequencies (DF) pre váženie dotazu
            data  # CSV dáta
        )

    except FileNotFoundError as e:
        print(
            f"CHYBA: Súbor {e.filename} nebol nájdený. Uistite sa, že ste spustili indexer a že súbory sú v adresári '{INDEX_DIR}'.")
        sys.exit(1)
    except Exception as e:
        print(f"CHYBA: Nepodarilo sa načítať dáta: {e}")
        sys.exit(1)


# Boolean Queries (AND, OR, NOT)
def get_all_doc_ids(meta_data):
    # Vráti množinu všetkých ID dokumentov ako reťazce
    return set(meta_data.keys())

# Parsovanie dopytu na zoznam (operátor, termín)
def parse_boolean_query_string(q):
    tokens = re.findall(r'"([^"]*)"|\b(AND|OR|NOT)\b|(\S+)', q, re.IGNORECASE)

    parsed = []
    current_op = "AND"

    for quoted, operator, unquoted in tokens:
        term_raw = quoted or unquoted

        # Spracovanie operátora
        if operator:
            current_op = operator.upper()
            continue

        # Spracovanie termínu
        if term_raw:
            term_raw = term_raw.strip()

            # Tokenizácia termínu, aby sa zabezpečila zhoda s indexom (napr. "5-10" ostane 5-10)
            # Ak je termín v úvodzovkách, tokenizuje sa len raz.
            term_tokens = tokenize(term_raw)
            term = term_tokens[0] if term_tokens else term_raw.lower()

            if not term:  # Vynechanie prázdnych tokenov po tokenizácii
                continue

            parsed.append((current_op, term))

            current_op = "AND"  # Reset operátora na implicitný AND

    return parsed

# Vyhodnotenie boolean dopytu. Vracia zoznam str doc_id
def boolean_query(parsed_terms, postings_index, meta_data, data):
    if not parsed_terms:
        return []

    universe = get_all_doc_ids(meta_data)
    result = None

    for op, term in parsed_terms:

        # Získanie Doc ID pre termín a jeho prípadná negácia
        postings = set(postings_index.get(term, []))  # str doc_id

        if op == "NOT":
            # Ak je operátor NOT, použije sa negovaná množina
            current_set = universe - postings
        else:
            # Inak sa použije normálny postings list
            current_set = postings

        # Spracovanie prvej operácie (Vždy nastavenie výsledku)
        if result is None:
            # Prvá operácia (či už AND, OR, alebo NOT) vždy inicializuje výsledok
            result = current_set

        # Spracovanie nasledujúcich operácií
        else:
            if op == "AND" or op == "NOT":  # NOT sa správa ako AND negovanej množiny
                # Prienik: Musí obsahovať predchádzajúci výsledok AND aktuálnu množinu
                result = result & current_set

            elif op == "OR":
                # Zjednotenie: Obsahuje predchádzajúci výsledok OR aktuálnu množinu
                result = result | current_set

    return sorted(list(result), key=lambda x: int(x))

# Rankované vyhľadávanie: Cosine Similarity
def ranked_search(query, inverted_tfidf_index, norms, df_counts, meta_data, data, idf_method="standard", top_k=10):
    tokens = tokenize(query)
    if not tokens: return []

    N = len(meta_data)  # Celkový počet dokumentov

    # Výber správnej IDF funkcie
    if idf_method == "standard":
        idf_func = get_idf_standard
    elif idf_method == "probabilistic":
        idf_func = get_idf_probabilistic
    else:
        print(f"Neznáma IDF metóda: {idf_method}. Používam 'standard'.")
        idf_method = "standard"
        idf_func = get_idf_standard

    # Vytvorenie vektora dopytu (Q) a výpočet normy dopytu
    q_tf = {}
    for t in tokens: q_tf[t] = q_tf.get(t, 0) + 1

    q_vec = {}
    q_sq_sum = 0.0

    for term, tf in q_tf.items():
        df_val = df_counts.get(term, 0)

        # Výpočet IDF váhy dopytu v reálnom čase
        idf_val = idf_func(df_val, N)

        tf_w = tf_weight(tf)
        q_w = tf_w * idf_val
        q_vec[term] = q_w
        q_sq_sum += q_w ** 2

    q_norm = math.sqrt(q_sq_sum)
    if q_norm == 0.0: return []

    # Identifikácia kandidátskych dokumentov
    # Používa sa invertovaný TF-IDF index (Term-to-Doc)
    candidate_docs = set()
    term_to_doc_weights = inverted_tfidf_index.get(idf_method, {})

    for term in q_vec:
        # Získanie všetkých doc_ids, ktoré obsahujú termín s príslušnou metódou váženia
        candidate_docs.update(term_to_doc_weights.get(term, {}).keys())

    final_candidate_docs = candidate_docs

    # Výpočet skóre (Cosine Similarity)
    scores = {}

    for doc_id in final_candidate_docs:
        # Normy sú uložené ako norms[method][doc_id]
        doc_norm = norms.get(idf_method, {}).get(doc_id, 0.0)

        if doc_norm == 0.0: continue

        dot_product = 0.0
        for term, q_w in q_vec.items():
            d_w = term_to_doc_weights.get(term, {}).get(doc_id, 0.0)
            dot_product += q_w * d_w

        score = dot_product / (q_norm * doc_norm)
        scores[doc_id] = score  # doc_id je str

    # Zoradenie výsledkov
    # Výsledky sa triedia ako (str_doc_id, score) páry
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


# Zjednotený výstup pre zobrazenie výsledkov (ranked aj boolean)
def display_results(results, data, title, is_ranked=False, top_k=10):
    total_found = len(results)
    print(f"\n=== {title} ===")
    print(f"Celkovo nájdených záznamov: {total_found}")

    if not results:
        print("Neboli nájdené žiadne výsledky.")
        return

    display_count = min(total_found, top_k)

    # Definícia stĺpcov, ich šírok a kľúčov z CSV
    COLUMN_MAPPING = [
        # Zobrazená hlavička | Šírka | Zarovnanie | Kľúč v dátach (alebo None pre špeciálne hodnoty)
        ("#", 3, '<', None),
        ("Doc ID", 6, '>', None),
        ("Score", 7, '>', None),
        ("Player", 22, '<', "Player"),
        ("Pos", 3, '<', "Position"),
        ("H", 4, '>', "Height"),
        ("W", 6, '>', "Weight"),
        ("Born", 4, '>', "BornYear"),
        ("City", 10, '<', "BornCity"),
        ("St/Ctry", 15, '<', "BornState"),
        ("Team", 7, '<', "CurrentTeam"),
        ("Died", 4, '>', "DiedYear"),
        ("College", 12, '<', "College"),
        ("Draft", 15, '<', "DraftTeam"),
        ("D.Yr", 4, '>', "DraftYear"),
    ]

    # Príprava hlavičky a oddeľovača
    header_parts = [f"{col[0].center(col[1])}" for col in COLUMN_MAPPING]
    header = "| " + " | ".join(header_parts) + " |"

    separator = "-" * (len(header) + 1)

    print(f"\n{separator}")
    print(header)
    print(separator)

    for i, item in enumerate(results[:display_count], 1):
        if is_ranked:
            doc_id, score = item
            score_display = f"{score:.4f}"
        else:
            doc_id = item
            score_display = "Boolean"  # Text pre boolean match

        row = data.get(doc_id)
        if not row: continue

        # Zostavenie riadka dát
        row_parts = []

        for index, (col_name, width, align, data_key) in enumerate(COLUMN_MAPPING):
            value = ""

            # Špeciálne hodnoty (index 0, 1, 2)
            if index == 0:  # #
                value = str(i)
            elif index == 1:  # Doc ID
                value = str(doc_id)
            elif index == 2:  # Score
                value = score_display
            else:
                # Ostatné stĺpce z dát
                value = str(row.get(data_key, "")).strip()

            # Robustné skrátenie, ak je text príliš dlhý
            if len(value) > width:
                value = value[:width]

            format_str = f"{{:{align}{width}}}"
            row_parts.append(format_str.format(value))

        print("| " + " | ".join(row_parts) + " |")

    print(separator)

    if total_found > display_count:
        print(f"\n...Zobrazených len prvých {display_count} z {total_found} celkových výsledkov.")

# Hlavný interaktívny cyklus
if __name__ == "__main__":
    try:
        postings_index, inverted_tfidf_index, norms, meta_data, df_counts, data = load_data()
    except SystemExit:
        sys.exit(1)

    print("\n--- Spustený interaktívny vyhľadávač ---")
    print("Dostupné IDF metódy pre ranking: standard, probabilistic")
    print("Pre ukončenie zadajte 'exit' alebo 'quit'.")

    # Hlavný cyklus
    while True:
        q_input = input("\nZadajte hľadaný dotaz: ").strip()

        if q_input.lower() in ["exit", "quit", "ukonci"]:
            print("Ukončujem vyhľadávač.")
            break

        if not q_input: continue

        original_query = q_input

        # Detekcia, či ide o boolean dopyt
        is_boolean_query = any(tok.upper() in ("AND", "OR", "NOT") for tok in original_query.split())

        # Vstup pre počet výsledkov
        try:
            k = int(input("Počet výsledkov na zobrazenie (predvolené 10): ").strip() or "10")
        except ValueError:
            k = 10

        # Spustenie vyhľadávania
        if is_boolean_query:
            # Boolean vyhľadávanie
            print("\n[*] Spúšťam Boolean Query...")
            parsed = parse_boolean_query_string(original_query)
            result_docids = boolean_query(parsed, postings_index, meta_data, data)

            display_results(
                result_docids,
                data,
                title=f"Boolean výsledky pre '{original_query}'",
                is_ranked=False,
                top_k=k
            )

        else:
            # Rankované vyhľadávanie
            m = input("IDF metóda (standard/probabilistic, predvolené standard): ").strip().lower() or "standard"

            print(f"[*] Spúšťam Rankovaný dotaz s metódou '{m}'...")

            ranked_results = ranked_search(original_query, inverted_tfidf_index, norms, df_counts, meta_data, data, m,
                                           k)

            display_results(
                ranked_results,
                data,
                title=f"Rankované výsledky pre '{original_query}' (IDF={m})",
                is_ranked=True,
                top_k=k
            )
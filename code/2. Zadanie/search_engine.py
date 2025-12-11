import lucene
from java.nio.file import Paths
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.search import IndexSearcher, BooleanClause, BooleanQuery, TermQuery
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.standard import StandardAnalyzer

# Inicializácia Lucene VM
def init_lucene_vm():
    if not lucene.getVMEnv():
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        print("Lucene VM initialized")

init_lucene_vm()

class NFLSearchEngine:
    def __init__(self, index_path="nfl_players_index"):
        print("Opening index:", index_path)
        directory = FSDirectory.open(Paths.get(index_path))
        reader = DirectoryReader.open(directory)
        self.searcher = IndexSearcher(reader)
        self.analyzer = StandardAnalyzer()
        print(f"Loaded index with {reader.numDocs()} documents")

        # Text fields pre full-text search
        self.search_fields = [
            "full_name_search", "position_search", "college_search", "location_search",
            "player", "position", "college", "currentteam", "draftteam",
            "highschool", "info", "borncity", "bornstate", "birthplace_other"
        ]

        # Boosty pre jednotlivé polia
        self.boosts = {
            "full_name_search": 10.0,
            "player": 9.0,
            "position_search": 8.0,
            "position": 7.0,
            "college_search": 1.0,
            "college": 1.0,
            "highschool": 1.0,
            "location_search": 1.0,
            "borncity": 1.5,
            "bornstate": 1.5,
            "birthplace_other": 1.5,
            "currentteam": 7.0,
            "draftteam": 2.5,
            "info": 0.1,
        }

        # Polia pre zobrazenie
        self.display_fields = [
            "player", "position", "college", "currentteam", "draftteam",
            "borncity", "bornstate", "bornyear", "draftyear",
            "height", "weight", "draft_round"
        ]

    def search(self, query_text, limit=10):
        if not query_text.strip():
            return [], 0

        query_text = query_text.strip()
        builder = BooleanQuery.Builder()

        def is_height_query(q):
            return "-" in q and all(c.isdigit() or c == "-" for c in q)

        def is_weight_query(q):
            return q.isdigit()

        if " NOT " in query_text.upper():
            parts = query_text.upper().split(" NOT ", 1)
            positive_part = parts[0].strip()
            negative_part = parts[1].strip()

            if " AND " in positive_part:
                q1_text, q2_text = [term.strip() for term in positive_part.split(" AND ", 1)]
                positive_builder = BooleanQuery.Builder()

                if is_weight_query(q1_text):
                    positive_builder.add(TermQuery(Term("weight", q1_text)), BooleanClause.Occur.SHOULD)
                elif is_height_query(q1_text):
                    positive_builder.add(TermQuery(Term("height", q1_text)), BooleanClause.Occur.SHOULD)
                else:
                    for field in self.search_fields:
                        boost = self.boosts.get(field, 1.0)
                        qp = QueryParser(field, self.analyzer)
                        try:
                            q = qp.parse(f"({q1_text})^{boost}")
                            positive_builder.add(q, BooleanClause.Occur.SHOULD)
                        except Exception:
                            continue

                if is_weight_query(q2_text):
                    positive_builder.add(TermQuery(Term("weight", q2_text)), BooleanClause.Occur.SHOULD)
                elif is_height_query(q2_text):
                    positive_builder.add(TermQuery(Term("height", q2_text)), BooleanClause.Occur.SHOULD)
                else:
                    for field in self.search_fields:
                        boost = self.boosts.get(field, 1.0)
                        qp = QueryParser(field, self.analyzer)
                        try:
                            q = qp.parse(f"({q2_text})^{boost}")
                            positive_builder.add(q, BooleanClause.Occur.SHOULD)
                        except Exception:
                            continue

                builder.add(positive_builder.build(), BooleanClause.Occur.MUST)
            else:
                if is_weight_query(positive_part):
                    builder.add(TermQuery(Term("weight", positive_part)), BooleanClause.Occur.MUST)
                elif is_height_query(positive_part):
                    builder.add(TermQuery(Term("height", positive_part)), BooleanClause.Occur.MUST)
                else:
                    positive_builder = BooleanQuery.Builder()
                    for field in self.search_fields:
                        boost = self.boosts.get(field, 1.0)
                        qp = QueryParser(field, self.analyzer)
                        try:
                            q = qp.parse(f"({positive_part})^{boost}")
                            positive_builder.add(q, BooleanClause.Occur.SHOULD)
                        except Exception:
                            continue
                    builder.add(positive_builder.build(), BooleanClause.Occur.MUST)

            # Buildovanie NOT query
            negative_builder = BooleanQuery.Builder()
            if is_weight_query(negative_part):
                negative_builder.add(TermQuery(Term("weight", negative_part)), BooleanClause.Occur.SHOULD)
            elif is_height_query(negative_part):
                negative_builder.add(TermQuery(Term("height", negative_part)), BooleanClause.Occur.SHOULD)
            else:
                for field in self.search_fields:
                    boost = self.boosts.get(field, 1.0)
                    qp = QueryParser(field, self.analyzer)
                    try:
                        q = qp.parse(f"({negative_part})^{boost}")
                        negative_builder.add(q, BooleanClause.Occur.SHOULD)
                    except Exception:
                        continue

            # Negatívna query na vynechanie nežiaducich outputov
            builder.add(negative_builder.build(), BooleanClause.Occur.MUST_NOT)

        # AND logika
        elif " AND " in query_text.upper():
            q1_text, q2_text = [term.strip() for term in query_text.upper().split(" AND ", 1)]
            q1_builder = BooleanQuery.Builder()
            q2_builder = BooleanQuery.Builder()

            # Podquery 1
            if is_weight_query(q1_text):
                q1_builder.add(TermQuery(Term("weight", q1_text)), BooleanClause.Occur.SHOULD)
            elif is_height_query(q1_text):
                q1_builder.add(TermQuery(Term("height", q1_text)), BooleanClause.Occur.SHOULD)
            else:
                for field in self.search_fields:
                    boost = self.boosts.get(field, 1.0)
                    qp = QueryParser(field, self.analyzer)
                    try:
                        q = qp.parse(f"({q1_text})^{boost}")
                        q1_builder.add(q, BooleanClause.Occur.SHOULD)
                    except Exception:
                        continue

            # Podquery 2
            if is_weight_query(q2_text):
                q2_builder.add(TermQuery(Term("weight", q2_text)), BooleanClause.Occur.SHOULD)
            elif is_height_query(q2_text):
                q2_builder.add(TermQuery(Term("height", q2_text)), BooleanClause.Occur.SHOULD)
            else:
                for field in self.search_fields:
                    boost = self.boosts.get(field, 1.0)
                    qp = QueryParser(field, self.analyzer)
                    try:
                        q = qp.parse(f"({q2_text})^{boost}")
                        q2_builder.add(q, BooleanClause.Occur.SHOULD)
                    except Exception:
                        continue

            builder.add(q1_builder.build(), BooleanClause.Occur.MUST)
            builder.add(q2_builder.build(), BooleanClause.Occur.MUST)

        else:  # OR logika
            for field in self.search_fields:
                boost = self.boosts.get(field, 1.0)
                qp = QueryParser(field, self.analyzer)
                try:
                    # Číselné query pre váhu alebo výšku
                    if is_weight_query(query_text):
                        q = TermQuery(Term("weight", query_text))
                    elif is_height_query(query_text):
                        q = TermQuery(Term("height", query_text))
                    else:
                        q = qp.parse(f"({query_text})^{boost}")
                    builder.add(q, BooleanClause.Occur.SHOULD)
                except Exception:
                    continue

        final_query = builder.build()
        top_docs = self.searcher.search(final_query, limit)
        total_hits = top_docs.totalHits.value()

        clean_results = []
        for sdoc in top_docs.scoreDocs:
            doc = self.searcher.storedFields().document(sdoc.doc)
            result_dict = {
                "score": float(f"{sdoc.score:.4f}"),
                "player": doc.get("player") or doc.get("full_name_search") or "Unknown",
            }

            for field in self.display_fields:
                if field == "player":
                    continue
                elif field == "position":
                    result_dict["position"] = doc.get("position") or doc.get("position_search") or "N/A"
                elif field == "college":
                    result_dict["college"] = doc.get("college") or doc.get("college_search") or "N/A"
                else:
                    result_dict[field] = doc.get(field) or "N/A"

            clean_results.append(result_dict)

        return clean_results, total_hits

    def _search_numeric_or_height_weight(self, query_text, limit):
        builder = BooleanQuery.Builder()

        # Exact match na StringField "height" a "weight"
        for field in ["height", "weight"]:
            tq = TermQuery(Term(field, query_text))
            builder.add(tq, BooleanClause.Occur.SHOULD)

        final_query = builder.build()
        top_docs = self.searcher.search(final_query, limit)
        return self._collect_results(top_docs)

    def _build_or_query(self, text):
        builder = BooleanQuery.Builder()
        for field in self.search_fields:
            boost = self.boosts.get(field, 1.0)
            qp = QueryParser(field, self.analyzer)
            try:
                q = qp.parse(f"({text})^{boost}")
                builder.add(q, BooleanClause.Occur.SHOULD)
            except Exception:
                continue
        return builder.build()

    def _collect_results(self, top_docs):
        total_hits = top_docs.totalHits.value()
        clean_results = []
        for sdoc in top_docs.scoreDocs:
            doc = self.searcher.storedFields().document(sdoc.doc)
            result = {"score": float(f"{sdoc.score:.4f}"),
                      "player": doc.get("player") or doc.get("full_name_search") or "Unknown"}
            for field in self.display_fields:
                if field == "player":
                    continue
                result[field] = doc.get(field) or "N/A"
            clean_results.append(result)
        return clean_results, total_hits

def display_search_results(clean_results, total_found, limit=10):
    display_count = min(len(clean_results), limit)
    COLUMN_MAPPING = [
        ("#", 3, '<', None), ("Score", 7, '>', None), ("Player", 20, '<', "player"),
        ("Pos", 3, '<', "position"), ("College", 12, '<', "college"),
        ("Team", 10, '<', "currentteam"), ("H", 4, '>', "height"),
        ("W", 6, '>', "weight"), ("City", 10, '<', "borncity"),
        ("St/Ctry", 5, '<', "bornstate"), ("D.Yr", 5, '<', "draftyear")
    ]

    print(f"\n=== NFL Search Results ===")
    print(f"Total matching documents: {total_found}")

    if not clean_results:
        print("No results found.")
        return

    header_parts = [f"{col[0].center(col[1])}" for col in COLUMN_MAPPING]
    header = "| " + " | ".join(header_parts) + " |"
    separator = "-" * len(header)
    print(f"\n{separator}")
    print(header)
    print(separator)

    for i, r in enumerate(clean_results[:display_count], 1):
        row_parts = []
        for idx, (col_name, width, align, data_key) in enumerate(COLUMN_MAPPING):
            value = ""
            if idx == 0:
                value = str(i)
            elif idx == 1:
                value = f"{r['score']:.4f}"
            else:
                value = str(r.get(data_key, "")).strip()
            if len(value) > width:
                value = value[:width]
            row_parts.append(f"{{:{align}{width}}}".format(value))
        print("| " + " | ".join(row_parts) + " |")
    print(separator)
    if total_found > display_count:
        print(f"\n...Displayed only the first {display_count} of {total_found} total results.")

def main():
    try:
        engine = NFLSearchEngine()
    except Exception as e:
        print(f"Error initializing search engine: {e}")
        return

    print("\n=== NFL Search ===\nType query\nType 'exit' to quit\n")

    while True:
        query = input("Search> ").strip()
        if query.lower() == "exit":
            break
        results, total_hits = engine.search(query)
        display_search_results(results, total_hits, limit=10)

if __name__ == "__main__":
    main()
import os
import csv
import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StringField, FieldType
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import FSDirectory
from datetime import datetime

# Inicializácia
def init_lucene_vm():
    if not lucene.getVMEnv():
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        print("Lucene VM initialized")

init_lucene_vm()

# Text Field: full-text search
TEXT_FT = FieldType()
TEXT_FT.setStored(True)
TEXT_FT.setTokenized(True)
TEXT_FT.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
TEXT_FT.freeze()

# String Field: exact matching/filering
STRING_FT = FieldType()
STRING_FT.setStored(True)
STRING_FT.setTokenized(False)
STRING_FT.setIndexOptions(IndexOptions.DOCS)
STRING_FT.freeze()

class NFLPlayersIndexer:
    def __init__(self, index_dir="nfl_players_index"):
        self.index_dir = index_dir

        self.analyzer = StandardAnalyzer()

        # Vytvorenie indexu pokiaľ už neexistuje
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

    def read_csv_file(self, csv_file_path):
        data = []
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    cleaned_row = {k.strip(): v for k, v in row.items()}
                    data.append(cleaned_row)
            print(f"Loaded {len(data)} records from {csv_file_path}")
            return data
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []

    def _get_field_type(self, column_lower):
        # Polia pre full-text search (tokenizované)
        text_columns = {
            'player', 'info', 'position', 'borncity', 'bornstate',
            'birthplace_other', 'currentteam', 'draftteam', 'college', 'highschool'
        }

        # Polia pre exact matching (netokenizované)
        string_columns = {
            'bornyear', 'draftyear', 'diedyear', 'height', 'weight',
            'draft_round', 'doc_id', 'indexed_at'
        }

        legacy_text_columns = {'name', 'nfl_player_name'}
        legacy_string_columns = {'birth_year'}

        if column_lower in text_columns or column_lower in legacy_text_columns:
            return TEXT_FT
        elif column_lower in string_columns or column_lower in legacy_string_columns:
            return STRING_FT
        else:
            print(f"Warning: Unknown column '{column_lower}' defaulting to STRING_FT")
            return STRING_FT

    def create_document_from_row(self, row, index):
        doc = Document()

        # Document ID a Timestamp
        doc.add(StringField("doc_id", str(index), Field.Store.YES))
        doc.add(StringField("indexed_at", datetime.now().isoformat(), Field.Store.YES))

        for column, value in row.items():
            value = value if value is not None else ""
            column_lower = column.lower()
            field_type = self._get_field_type(column_lower)

            # Čistá hodnota pre weight
            if column_lower == "weight":
                # Odstrániť všetko okrem číslic
                numeric_weight = ''.join(c for c in str(value) if c.isdigit())
                doc.add(Field("weight", numeric_weight, STRING_FT))
                doc.add(Field("weight_text", numeric_weight, TEXT_FT))
            # Pre height necháme formát 6-0 alebo 5-11
            elif column_lower == "height":
                doc.add(Field("height", str(value), STRING_FT))
                doc.add(Field("height_text", str(value), TEXT_FT))
            else:
                doc.add(Field(column_lower, str(value), field_type))

        # Full Name Search
        name = row.get('Player', '') or row.get('name', '') or row.get('player_name', '') or row.get('nfl_player_name',
                                                                                                     '')
        if name:
            doc.add(Field("full_name_search", name, TEXT_FT))

        # Position Search
        position = row.get('Position', '')
        if position:
            doc.add(Field("position_search", position, TEXT_FT))

        # College Search
        college = row.get('College', '')
        if college:
            doc.add(Field("college_search", college, TEXT_FT))

        # Location Search
        city = row.get('BornCity', '') or ''
        state = row.get('BornState', '') or ''
        other = row.get('BirthPlace_Other', '') or ''
        location_search = ' '.join(filter(None, [city, state, other]))
        if location_search:
            doc.add(Field("location_search", location_search, TEXT_FT))

        return doc

    def create_index(self, data, writer):
        print("Starting index creation...")

        indexed_count = 0
        # Indexovanie každého dokumentu
        for index, row in enumerate(data):
            try:
                doc = self.create_document_from_row(row, index)

                writer.addDocument(doc)
                indexed_count += 1

                if indexed_count % 1000 == 0:
                    print(f"Indexed {indexed_count} documents...")

            except Exception as e:
                print(f"Error indexing document {index}: {e}")
                continue

        return indexed_count

    def index_data_source(self, csv_file_path):
        print(f"Loading data from {csv_file_path}...")

        data = self.read_csv_file(csv_file_path)
        if not data:
            print("No data loaded from CSV file!")
            return 0

        if data:
            columns = list(data[0].keys())
            print(f"Columns: {columns}")
            print("Sample data (first 3 records):")
            for i in range(min(3, len(data))):
                print(f"  {data[i]}")

        store = FSDirectory.open(Paths.get(self.index_dir))
        config = IndexWriterConfig(self.analyzer)

        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
        writer = IndexWriter(store, config)

        try:
            indexed_count = self.create_index(data, writer)
        finally:
            writer.commit()
            writer.close()

        print(f"Index creation completed! Indexed {indexed_count} documents.")
        return indexed_count

    def create_index_from_csv(self, csv_file_path):
        return self.index_data_source(csv_file_path)

    def create_index_from_joined_data(self, joined_csv_path):
        print(f"Indexing joined NFL players data from {joined_csv_path}...")
        return self.index_data_source(joined_csv_path)

def find_csv_files():
    csv_files = []
    for file in os.listdir(os.getcwd()):
        if file.endswith(".csv"):
            csv_files.append(file)
    return csv_files

def main():
    print("NFL Players Lucene Indexer (Indexing Only)")
    print("=" * 50)

    # Inicializácia indexera
    indexer = NFLPlayersIndexer("nfl_players_index")

    joined_csv = "joined_data.csv"
    csv_files = find_csv_files()

    index_exists = os.path.exists("nfl_players_index") and len(os.listdir("nfl_players_index")) > 0

    if index_exists:
        response = input("Index already exists. Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Index creation cancelled.")
            return

    print("Creating new index...")

    if os.path.exists(joined_csv):
        print(f"Found joined data file: {joined_csv}")
        count = indexer.create_index_from_joined_data(joined_csv)
    elif csv_files:
        print(f"Found CSV files: {csv_files}")
        count = indexer.create_index_from_csv(csv_files[0])
    else:
        print("No CSV files found in current directory!")
        return

    print(f"\nIndex creation completed successfully!")
    print(f"Total documents indexed: {count}")

if __name__ == "__main__":
    main()
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, lit, trim, upper, coalesce, when
from pyspark.sql.types import StringType
from pyspark.sql.functions import asc

# Inicializácia Sparku
spark = SparkSession.builder \
    .appName("NFLPlayersJoinFinalSort") \
    .getOrCreate()

# Čítanie CSV
nfl_players_df = spark.read.csv("nfl_players.csv", header=True, inferSchema=True)
other_players_df = spark.read.csv("part-00000-d89e5604-8460-4301-9b50-7ba69571ceac-c000.csv", header=True,
                                  inferSchema=True)

# Castnutie všetkých columns ako StringType
def cast_all_to_string(df):
    """Casts all columns in a DataFrame to StringType."""
    for column_name in df.columns:
        df = df.withColumn(column_name, col(column_name).cast(StringType()))
    return df

nfl_players_df = cast_all_to_string(nfl_players_df)
other_players_df = cast_all_to_string(other_players_df)

# Očistenie dát pre lepšie matchovanie
nfl_players_df = nfl_players_df.withColumn("Player_clean", trim(upper(col("Player"))))
other_players_df = other_players_df.withColumn("player_name_clean", trim(upper(col("player_name"))))

# Kombinovanie polí výšky (height_ft a height_in), spracovanie NULL/N/A hodnôt
other_players_df = other_players_df.withColumn(
    "Height_combined",
    when(
        (col("height_ft").isNull() | (upper(trim(col("height_ft"))) == lit("N/A"))) |
        (col("height_in").isNull() | (upper(trim(col("height_in"))) == lit("N/A"))),
        lit("N/A")
    ).otherwise(
        concat_ws("-", col("height_ft"), col("height_in"))
    )
)

# Premenovanie columns
other_players_df = other_players_df \
    .withColumnRenamed("birth_year", "BornYear") \
    .withColumnRenamed("birth_place", "BirthPlace") \
    .withColumnRenamed("weight_lb", "Weight") \
    .withColumnRenamed("college", "College") \
    .withColumnRenamed("high_school", "HighSchool") \
    .withColumnRenamed("draft_year", "DraftYear")

# Vykonanie outer joinu použitím mena a roku narodenia
outer_join_df = nfl_players_df.alias("nfl").join(
    other_players_df.alias("other"),
    (
            (col("nfl.Player_clean") == col("other.player_name_clean")) &
            (col("nfl.BornYear") == col("other.BornYear"))
    ),
    "outer"
)

# Mergovanie a čistenie
def merge_and_clean(col_nfl, col_other, alias_name):
    merged_col = coalesce(col_nfl, col_other)
    return coalesce(merged_col, lit("N/A")).alias(alias_name)

def clean_null_to_na(spark_col, alias_name):
    return coalesce(spark_col, lit("N/A")).alias(alias_name)

player_merged_cleaned = merge_and_clean(col("nfl.Player"), col("other.player_name"), "Player")
height_merged_cleaned = merge_and_clean(col("nfl.Height"), col("other.Height_combined"), "Height")
weight_merged_cleaned = merge_and_clean(col("nfl.Weight"), col("other.Weight"), "Weight")
born_year_merged_cleaned = merge_and_clean(col("nfl.BornYear"), col("other.BornYear"), "BornYear")
draft_year_merged_cleaned = merge_and_clean(col("nfl.DraftYear"), col("other.DraftYear"), "DraftYear")
college_merged_cleaned = merge_and_clean(col("nfl.College"), col("other.College"), "College")
high_school_merged_cleaned = merge_and_clean(col("nfl.HighSchool"), col("other.HighSchool"), "HighSchool")

final_df = outer_join_df.select(
    player_merged_cleaned,
    height_merged_cleaned,
    weight_merged_cleaned,
    born_year_merged_cleaned,
    draft_year_merged_cleaned,
    college_merged_cleaned,
    high_school_merged_cleaned,

    clean_null_to_na(col("nfl.Position"), "Position"),
    clean_null_to_na(col("nfl.BornCity"), "BornCity"),
    clean_null_to_na(col("nfl.BornState"), "BornState"),
    clean_null_to_na(col("other.BirthPlace"), "BirthPlace_Other"),
    clean_null_to_na(col("nfl.CurrentTeam"), "CurrentTeam"),
    clean_null_to_na(col("nfl.DraftTeam"), "DraftTeam"),
    clean_null_to_na(col("other.draft_round"), "draft_round"),
    clean_null_to_na(col("nfl.DiedYear"), "DiedYear"),
    clean_null_to_na(col("other.info"), "info")
)

# Sortnutie od A po Z
final_df_sorted = final_df.sort(asc("Player"))

# Uloženie ako jedno CSV
final_df_sorted.coalesce(1).write.csv(
    "nfl_players_merged_sorted_single.csv",
    header=True,
    mode="overwrite"
)

print("File saved as: nfl_players_merged_sorted_single.csv")

spark.stop()
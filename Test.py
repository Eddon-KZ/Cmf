# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, lit, months_between
from pyspark.sql.window import Window

# Init Spark
spark = SparkSession.builder.getOrCreate()

# Read inputs (Dataiku pandas -> Spark)
ter_df = spark.createDataFrame(contract_TER_df)
mel_df = spark.createDataFrame(contract_mel_from_df)


# =========================
# FUNCTION BUILD KPI
# =========================
def build_ter_contracts_with_loyalty_KPIs(ter_df, ter_mel_df):

    renewal_timing_min = -1

    # Keep only matched rows (équivalent _merge == 'both')
    ter_renewed_df = ter_mel_df.filter(col("Contract_Date_MEL").isNotNull())

    # Compute renewal timing
    ter_renewed_df = ter_renewed_df.withColumn(
        "renewal_timing_in_months",
        months_between(col("Contract_Date_MEL"), col("Contract_Date_TER")).cast("int")
    )

    # Filter timing
    ter_renewed_df = ter_renewed_df.filter(
        col("renewal_timing_in_months") >= renewal_timing_min
    )

    # is_loyal
    ter_renewed_df = ter_renewed_df.withColumn("is_loyal", lit(1))

    # same manufacturer
    ter_renewed_df = ter_renewed_df.withColumn(
        "is_same_manufacturer",
        when(col("Manufacturer") == col("Manufacturer_stl"), 1).otherwise(0)
    )

    # nb loyalty times
    window_spec = Window.partitionBy("PK_Lots")
    ter_renewed_df = ter_renewed_df.withColumn(
        "nb_loyalty_times",
        count("PK_Lots").over(window_spec)
    )

    # NOT renewed contracts
    renewed_pk = ter_renewed_df.select("PK_Lots").distinct()

    ter_not_renewed_df = ter_df.join(
        renewed_pk,
        on="PK_Lots",
        how="left_anti"
    ).withColumn("is_loyal", lit(0)) \
     .withColumn("nb_loyalty_times", lit(0))

    # concat
    ter_contracts = ter_renewed_df.unionByName(ter_not_renewed_df)

    # fillna
    ter_contracts = ter_contracts.fillna({"Client_Siren": 0})

    return ter_contracts


# =========================
# MAIN FUNCTION
# =========================
def define_customer_asset_loyalty(ter_df, mel_df):

    # Select useful columns
    mel_keep_df = mel_df.select(
        "PK_Lots",
        "Client_Siren",
        "Materiel_Type",
        "Country",
        "Contract_Date_MEL",
        "Manufacturer"
    )

    # JOIN (⚠️ toujours cartésien mais Spark supporte)
    df = ter_df.join(
        mel_keep_df,
        on=["Client_Siren", "Materiel_Type", "Country"],
        how="left"
    )

    # Rename MEL manufacturer to match pandas suffix
    df = df.withColumnRenamed("Manufacturer", "Manufacturer_stl")

    # Build KPIs
    ter_contracts = build_ter_contracts_with_loyalty_KPIs(ter_df, df)

    # rename
    ter_contracts = ter_contracts.withColumnRenamed(
        "is_loyal", "is_loyal_customer_asset"
    )

    return ter_contracts


# =========================
# RUN
# =========================
result_df = define_customer_asset_loyalty(ter_df, mel_df)

# Convert back to pandas for Dataiku write
result_pd = result_df.toPandas()

# Write output
output_dataset = dataiku.Dataset("SRW_contrat_ter_loyalty_siren_asset")
output_dataset.write_with_schema(result_pd)

# ==============================================================================
# SILVER LAYER ETL — credit-risk-silver-etl.py
# Tujuan : Bersihkan raw data, feature engineering per tabel,
#          agregasi per nasabah (SK_ID_CURR), simpan ke 02-silver/
# Sumber  : Tabel_ETL.xlsx — Domain Knowledge & Agregasi per Nasabah
# ==============================================================================

import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F

args        = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session
job         = Job(glueContext)
job.init(args["JOB_NAME"], args)

BUCKET      = "s3://credit-risk-checker-datalake-242046727318-us-east-1-an/"
INPUT_BASE  = BUCKET + "00-raw/credit-risk-checker/"
SILVER_BASE = BUCKET + "02-silver/"

print("[INFO] Spark session aktif. Memulai Silver ETL Pipeline...")

# ==============================================================================
# BLOCK 1: INGESTION
# ==============================================================================
app            = spark.read.csv(INPUT_BASE + "application_train.csv",     header=True, inferSchema=True)
app_test       = spark.read.csv(INPUT_BASE + "application_test.csv",      header=True, inferSchema=True)
bureau         = spark.read.csv(INPUT_BASE + "bureau.csv",                header=True, inferSchema=True)
bureau_balance = spark.read.csv(INPUT_BASE + "bureau_balance.csv",        header=True, inferSchema=True)
inst           = spark.read.csv(INPUT_BASE + "installments_payments.csv", header=True, inferSchema=True)
prev           = spark.read.csv(INPUT_BASE + "previous_application.csv",  header=True, inferSchema=True)
card           = spark.read.csv(INPUT_BASE + "credit_card_balance.csv",   header=True, inferSchema=True)
pos            = spark.read.csv(INPUT_BASE + "POS_CASH_balance.csv",      header=True, inferSchema=True)

print("[SUCCESS] Ingesti 8 tabel selesai.")

# ==============================================================================
# BLOCK 2: APPLICATION — Feature Engineering
# Sumber: application_train | Domain: Pendapatan vs hutang, rasio finansial
# ==============================================================================
def build_app_features(df):
    return (
        df
        # Sentinel value: DAYS_EMPLOYED = 365243 artinya pengangguran
        .withColumn("IS_UNEMPLOYED",
                    F.when(F.col("DAYS_EMPLOYED") == 365243, 1).otherwise(0))
        .withColumn("DAYS_EMPLOYED_YEARS",
                    F.when(F.col("DAYS_EMPLOYED") == 365243, None)
                     .otherwise(F.col("DAYS_EMPLOYED") / -365))

        # Jumlah external source yang tersedia
        .withColumn("EXT_SOURCE_COUNT",
                    F.when(F.col("EXT_SOURCE_1").isNotNull(), 1).otherwise(0) +
                    F.when(F.col("EXT_SOURCE_2").isNotNull(), 1).otherwise(0) +
                    F.when(F.col("EXT_SOURCE_3").isNotNull(), 1).otherwise(0))

        # Rata-rata external source (aman dari divide-by-zero & ANSI mode)
        .withColumn("EXT_SOURCE_MEAN",
                    F.when(F.col("EXT_SOURCE_COUNT") == 0, None)
                     .otherwise(
                        (F.coalesce(F.col("EXT_SOURCE_1"), F.lit(0)) +
                         F.coalesce(F.col("EXT_SOURCE_2"), F.lit(0)) +
                         F.coalesce(F.col("EXT_SOURCE_3"), F.lit(0))) / F.col("EXT_SOURCE_COUNT")
                    ))

        # Standar deviasi external source (butuh minimal 2 nilai)
        .withColumn("EXT_SOURCE_STD",
                    F.when(F.col("EXT_SOURCE_COUNT") >= 2,
                        F.sqrt(
                            (F.pow(F.coalesce(F.col("EXT_SOURCE_1"), F.col("EXT_SOURCE_MEAN")) - F.col("EXT_SOURCE_MEAN"), 2) +
                             F.pow(F.coalesce(F.col("EXT_SOURCE_2"), F.col("EXT_SOURCE_MEAN")) - F.col("EXT_SOURCE_MEAN"), 2) +
                             F.pow(F.coalesce(F.col("EXT_SOURCE_3"), F.col("EXT_SOURCE_MEAN")) - F.col("EXT_SOURCE_MEAN"), 2)) / 3
                        )
                    ).otherwise(None))

        # Fitur rasio finansial (Tabel_ETL: debt-to-income, annuity-to-income)
        .withColumn("debt_to_income",
                    F.col("AMT_CREDIT") / F.when(F.col("AMT_INCOME_TOTAL") != 0, F.col("AMT_INCOME_TOTAL")))
        .withColumn("annuity_to_income",
                    F.col("AMT_ANNUITY") / F.when(F.col("AMT_INCOME_TOTAL") != 0, F.col("AMT_INCOME_TOTAL")))
        .withColumn("GOODS_CREDIT_RATIO",
                    F.col("AMT_GOODS_PRICE") / F.when(F.col("AMT_CREDIT") != 0, F.col("AMT_CREDIT")))
        .withColumn("INCOME_PER_PERSON",
                    F.col("AMT_INCOME_TOTAL") / F.when(F.col("CNT_FAM_MEMBERS") != 0, F.col("CNT_FAM_MEMBERS")))
        .withColumn("EMPLOYED_TO_AGE_RATIO",
                    F.when((F.col("DAYS_EMPLOYED") != 365243) & (F.col("DAYS_BIRTH") != 0),
                           F.col("DAYS_EMPLOYED") / F.col("DAYS_BIRTH")).otherwise(None))
    )

app      = build_app_features(app)
app_test = build_app_features(app_test)
print("[SUCCESS] Application feature engineering selesai.")

# ==============================================================================
# BLOCK 3: BUREAU BALANCE — Agregasi per SK_ID_BUREAU
# Sumber: bureau_balance | Domain: Status bulanan kredit luar
# STATUS: 0=ok, 1-5=DPD (Days Past Due), C=closed, X=unknown
# ==============================================================================
bb_agg = bureau_balance.groupBy("SK_ID_BUREAU").agg(
    F.count("*").alias("bb_months_count"),
    F.sum(F.when(F.col("STATUS").isin(["1","2","3","4","5"]), 1).otherwise(0)).alias("bb_dpd_months"),
    F.avg(F.when(F.col("STATUS").isin(["1","2","3","4","5"]), 1).otherwise(0)).alias("bb_dpd_rate")
)

bureau = bureau.join(bb_agg, "SK_ID_BUREAU", "left")
print("[SUCCESS] Bureau balance agregasi selesai.")

# ==============================================================================
# BLOCK 4: BUREAU — Agregasi per SK_ID_CURR
# Sumber: bureau | Domain: Lama riwayat kredit, jumlah kredit aktif
# ==============================================================================
bureau_agg = bureau.groupBy("SK_ID_CURR").agg(
    F.count("*").alias("bureau_total_loans"),
    F.sum(F.when(F.col("CREDIT_ACTIVE") == "Active", 1).otherwise(0)).alias("bureau_active_count"),
    F.avg(F.col("DAYS_CREDIT") / -365).alias("bureau_avg_credit_age_years"),
    F.sum("AMT_CREDIT_SUM").alias("bureau_total_credit"),
    F.sum("AMT_CREDIT_SUM_DEBT").alias("bureau_total_debt"),
    # Carry forward bb agregasi
    F.avg("bb_dpd_rate").alias("bureau_avg_bb_dpd_rate"),
    F.sum("bb_dpd_months").alias("bureau_total_bb_dpd_months")
).na.fill(0, subset=[
    "bureau_total_loans", "bureau_active_count", "bureau_avg_credit_age_years",
    "bureau_total_credit", "bureau_total_debt",
    "bureau_avg_bb_dpd_rate", "bureau_total_bb_dpd_months"
])
print("[SUCCESS] Bureau agregasi selesai.")

# ==============================================================================
# BLOCK 5: INSTALLMENT PAYMENTS — Agregasi per SK_ID_CURR
# Sumber: installment_payment | Domain: Riwayat keterlambatan cicilan
# ==============================================================================
inst = (
    inst
    .withColumn("days_late", F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT"))
    .withColumn("paid_late",  F.when(F.col("days_late") > 0, 1).otherwise(0))
    .withColumn("amt_underpay",
                F.greatest(F.col("AMT_INSTALMENT") - F.col("AMT_PAYMENT"), F.lit(0)))
)

inst_agg = inst.groupBy("SK_ID_CURR").agg(
    F.count("*").alias("inst_total_payments"),
    F.avg("days_late").alias("inst_avg_days_late"),
    F.avg("paid_late").alias("inst_pct_late"),
    F.sum("paid_late").alias("inst_count_late"),
    F.avg("amt_underpay").alias("inst_avg_underpay")
).na.fill(0, subset=[
    "inst_total_payments", "inst_avg_days_late",
    "inst_pct_late", "inst_count_late", "inst_avg_underpay"
])
print("[SUCCESS] Installment payments agregasi selesai.")

# ==============================================================================
# BLOCK 6: CREDIT CARD BALANCE — Agregasi per SK_ID_CURR
# Sumber: credit_card_balance | Domain: Credit utilization ratio
# ==============================================================================
card = card.withColumn(
    "utilization",
    F.col("AMT_BALANCE") / F.when(F.col("AMT_CREDIT_LIMIT_ACTUAL") != 0, F.col("AMT_CREDIT_LIMIT_ACTUAL"))
)

card_agg = card.groupBy("SK_ID_CURR").agg(
    F.avg("utilization").alias("card_avg_utilization"),
    F.max("utilization").alias("card_max_utilization"),
    F.avg("AMT_BALANCE").alias("card_avg_balance"),
    F.avg(F.col("AMT_PAYMENT_CURRENT") / F.when(F.col("AMT_INST_MIN_REGULARITY") != 0,
          F.col("AMT_INST_MIN_REGULARITY"))).alias("card_avg_payment_ratio")
).na.fill(0, subset=[
    "card_avg_utilization", "card_max_utilization",
    "card_avg_balance", "card_avg_payment_ratio"
])
print("[SUCCESS] Credit card balance agregasi selesai.")

# ==============================================================================
# BLOCK 7: PREVIOUS APPLICATION — Agregasi per SK_ID_CURR
# Sumber: previous_application | Domain: Application rejection rate
# ==============================================================================
prev_agg = prev.groupBy("SK_ID_CURR").agg(
    F.count("*").alias("prev_app_count"),
    F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Approved", 1).otherwise(0)).alias("prev_approved_count"),
    F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Refused",  1).otherwise(0)).alias("prev_refused_count"),
    F.avg(F.when(F.col("NAME_CONTRACT_STATUS") == "Refused",  1).otherwise(0)).alias("prev_app_rejection_rate"),
    F.avg("AMT_CREDIT").alias("prev_avg_credit")
).na.fill(0, subset=[
    "prev_app_count", "prev_approved_count", "prev_refused_count",
    "prev_app_rejection_rate", "prev_avg_credit"
])
print("[SUCCESS] Previous application agregasi selesai.")

# ==============================================================================
# BLOCK 8: POS CASH BALANCE — Agregasi per SK_ID_CURR
# Sumber: POS_cash_balance | Domain: Beban pinjaman POS, proporsi keterlambatan
# ==============================================================================
pos_agg = pos.groupBy("SK_ID_CURR").agg(
    F.count("*").alias("pos_count_months"),
    F.avg(F.when(F.col("SK_DPD") > 0, 1).otherwise(0)).alias("pos_pct_dpd"),
    F.max("SK_DPD").alias("pos_max_dpd"),
    F.countDistinct("SK_ID_PREV").alias("pos_loan_count")
).na.fill(0, subset=[
    "pos_count_months", "pos_pct_dpd", "pos_max_dpd", "pos_loan_count"
])
print("[SUCCESS] POS cash balance agregasi selesai.")

# ==============================================================================
# BLOCK 9: MERGE & CROSS-TABLE FEATURE ENGINEERING
# ==============================================================================
def merge_silver(base_df):
    return (
        base_df
        .join(bureau_agg, "SK_ID_CURR", "left")
        .join(inst_agg,   "SK_ID_CURR", "left")
        .join(prev_agg,   "SK_ID_CURR", "left")
        .join(card_agg,   "SK_ID_CURR", "left")
        .join(pos_agg,    "SK_ID_CURR", "left")
        # [FIX] fillna hanya kolom numerik agregasi — hindari mengisi kolom string dengan "0"
        .na.fill(0, subset=[
            "bureau_total_loans", "bureau_active_count", "bureau_avg_credit_age_years",
            "bureau_total_credit", "bureau_total_debt",
            "bureau_avg_bb_dpd_rate", "bureau_total_bb_dpd_months",
            "inst_total_payments", "inst_avg_days_late", "inst_pct_late",
            "inst_count_late", "inst_avg_underpay",
            "prev_app_count", "prev_approved_count", "prev_refused_count",
            "prev_app_rejection_rate", "prev_avg_credit",
            "card_avg_utilization", "card_max_utilization",
            "card_avg_balance", "card_avg_payment_ratio",
            "pos_count_months", "pos_pct_dpd", "pos_max_dpd", "pos_loan_count"
        ])
        # Cross-table features
        .withColumn("bureau_debt_utilization",
                    F.col("bureau_total_debt") /
                    F.when(F.col("bureau_total_credit") != 0, F.col("bureau_total_credit")))
        .withColumn("total_debt_to_income",
                    F.col("bureau_total_debt") /
                    F.when(F.col("AMT_INCOME_TOTAL") != 0, F.col("AMT_INCOME_TOTAL")))
    )

df_train_silver = merge_silver(app)
df_test_silver  = merge_silver(app_test)

# ==============================================================================
# BLOCK 10: SIMPAN KE SILVER
# ==============================================================================
df_train_silver.repartition(20).write.mode("overwrite").parquet(SILVER_BASE + "train_features_v4/")
df_test_silver.repartition(10).write.mode("overwrite").parquet(SILVER_BASE + "test_features_v4/")

print("[SUCCESS] Silver layer tersimpan di S3!")
print(f"  Train : {SILVER_BASE}train_features_v4/")
print(f"  Test  : {SILVER_BASE}test_features_v4/")

job.commit()

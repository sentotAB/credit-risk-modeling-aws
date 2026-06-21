# ==============================================================================
# GOLD LAYER ETL — credit-risk-gold-etl.py
# Tujuan : Baca Silver, buat fitur lanjutan siap model (seleksi, binning,
#          interaksi fitur), simpan ke 03-gold/
# Input  : 02-silver/train_features_v4/ & test_features_v4/
# Output : 03-gold/train_features_gold/ & test_features_gold/
# ==============================================================================

import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

args        = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session
job         = Job(glueContext)
job.init(args["JOB_NAME"], args)

BUCKET      = "s3://credit-risk-checker-datalake-242046727318-us-east-1-an/"
SILVER_BASE = BUCKET + "02-silver/"
GOLD_BASE   = BUCKET + "03-gold/"

print("[INFO] Membaca data dari Silver layer...")
df_train = spark.read.parquet(SILVER_BASE + "train_features_v4/")
df_test  = spark.read.parquet(SILVER_BASE + "test_features_v4/")
print(f"[SUCCESS] Silver dimuat. Train: {df_train.count():,} baris | Test: {df_test.count():,} baris")

# ==============================================================================
# BLOCK 1: KOREKSI NILAI 0 PALSU
# Kolom-kolom ini NULL jika nasabah tidak punya riwayat — bukan 0
# (akibat bug fillna di ETL lama, dipastikan bersih di sini)
# ==============================================================================
COLS_NULL_WHEN_NO_HISTORY = [
    "bureau_avg_credit_age_years",  # null = belum pernah punya kredit luar
    "inst_avg_days_late",           # null = tidak ada riwayat cicilan
    "card_avg_utilization",         # null = tidak punya kartu kredit
    "card_max_utilization",
]

def fix_false_zeros(df):
    for col in COLS_NULL_WHEN_NO_HISTORY:
        if col in df.columns:
            df = df.withColumn(col, F.when(F.col(col) == 0, None).otherwise(F.col(col)))
    return df

df_train = fix_false_zeros(df_train)
df_test  = fix_false_zeros(df_test)
print("[SUCCESS] Koreksi nilai 0 palsu selesai.")

# ==============================================================================
# BLOCK 2: ADVANCED FEATURE ENGINEERING
# Fitur interaksi antar tabel yang tidak bisa dibuat di Silver
# ==============================================================================
def build_gold_features(df):
    return (
        df

        # ── RISIKO GABUNGAN ────────────────────────────────────────────────────
        # Skor risiko agregat: gabungkan sinyal keterlambatan dari semua sumber
        .withColumn("combined_late_signal",
                    (F.coalesce(F.col("inst_pct_late"),       F.lit(0)) +
                     F.coalesce(F.col("pos_pct_dpd"),         F.lit(0)) +
                     F.coalesce(F.col("bureau_avg_bb_dpd_rate"), F.lit(0))) / 3)

        # ── RASIO HUTANG TOTAL ─────────────────────────────────────────────────
        # Total beban hutang (kredit baru + hutang bureau) relatif terhadap income
        .withColumn("total_all_debt_to_income",
                    (F.coalesce(F.col("AMT_CREDIT"),        F.lit(0)) +
                     F.coalesce(F.col("bureau_total_debt"), F.lit(0))) /
                    F.when(F.col("AMT_INCOME_TOTAL") != 0, F.col("AMT_INCOME_TOTAL")))

        # ── KUALITAS RIWAYAT KREDIT ────────────────────────────────────────────
        # Rasio kredit aktif vs total kredit di bureau
        .withColumn("bureau_active_ratio",
                    F.col("bureau_active_count") /
                    F.when(F.col("bureau_total_loans") != 0, F.col("bureau_total_loans")))

        # ── BEBAN CICILAN ──────────────────────────────────────────────────────
        # Annuity relatif terhadap income per anggota keluarga
        .withColumn("annuity_per_person",
                    F.col("AMT_ANNUITY") /
                    F.when(F.col("CNT_FAM_MEMBERS") != 0, F.col("CNT_FAM_MEMBERS")))

        # Rasio underpayment cicilan (seberapa sering bayar kurang dari tagihan)
        .withColumn("inst_underpay_ratio",
                    F.col("inst_avg_underpay") /
                    F.when(F.col("AMT_ANNUITY") != 0, F.col("AMT_ANNUITY")))

        # ── KARTU KREDIT ───────────────────────────────────────────────────────
        # Flag nasabah dengan utilisasi kartu sangat tinggi (>80%)
        .withColumn("flag_high_card_util",
                    F.when(F.col("card_avg_utilization") > 0.8, 1).otherwise(0))

        # ── RIWAYAT PENGAJUAN KREDIT ───────────────────────────────────────────
        # Perbandingan kredit disetujui vs kredit baru yang diminta
        .withColumn("prev_credit_vs_current",
                    F.col("prev_avg_credit") /
                    F.when(F.col("AMT_CREDIT") != 0, F.col("AMT_CREDIT")))

        # ── BINNING FITUR KUNCI ────────────────────────────────────────────────
        # Bin debt-to-income menjadi 4 kelompok risiko (low/medium/high/very high)
        .withColumn("dti_risk_bin",
                    F.when(F.col("debt_to_income") < 1.0,  F.lit("low"))
                     .when(F.col("debt_to_income") < 2.0,  F.lit("medium"))
                     .when(F.col("debt_to_income") < 4.0,  F.lit("high"))
                     .otherwise(F.lit("very_high")))

        # Bin lama riwayat kredit di bureau
        .withColumn("credit_age_bin",
                    F.when(F.col("bureau_avg_credit_age_years").isNull(), F.lit("no_history"))
                     .when(F.col("bureau_avg_credit_age_years") < 1,      F.lit("new"))
                     .when(F.col("bureau_avg_credit_age_years") < 3,      F.lit("short"))
                     .when(F.col("bureau_avg_credit_age_years") < 7,      F.lit("medium"))
                     .otherwise(F.lit("long")))

        # ── FLAG RISIKO ────────────────────────────────────────────────────────
        # Nasabah berisiko tinggi: banyak penolakan + telat bayar + utilisasi tinggi
        .withColumn("flag_high_risk_profile",
                    F.when(
                        (F.col("prev_app_rejection_rate") > 0.3) &
                        (F.col("inst_pct_late") > 0.2) &
                        (F.col("card_avg_utilization") > 0.6),
                    1).otherwise(0))

        # Nasabah baru tanpa riwayat sama sekali
        .withColumn("flag_no_credit_history",
                    F.when(F.col("bureau_total_loans") == 0, 1).otherwise(0))
    )

df_train_gold = build_gold_features(df_train)
df_test_gold  = build_gold_features(df_test)
print("[SUCCESS] Gold feature engineering selesai.")

# ==============================================================================
# BLOCK 3: SELEKSI KOLOM FINAL UNTUK MODEL
# Hanya kolom yang relevan untuk modeling yang disimpan ke gold
# ==============================================================================
GOLD_FEATURES = [
    # Identifier & Target
    "SK_ID_CURR",
    "TARGET",                           # hanya ada di train

    # ── APPLICATION (fitur asli) ──────────────────────────────────────────────
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED_YEARS", "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "EXT_SOURCE_MEAN", "EXT_SOURCE_STD", "EXT_SOURCE_COUNT",
    "IS_UNEMPLOYED",

    # Kategorikal aplikasi
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",

    # ── RASIO FINANSIAL (Silver) ──────────────────────────────────────────────
    "debt_to_income", "annuity_to_income", "GOODS_CREDIT_RATIO",
    "INCOME_PER_PERSON", "EMPLOYED_TO_AGE_RATIO",

    # ── BUREAU ────────────────────────────────────────────────────────────────
    "bureau_total_loans", "bureau_active_count", "bureau_avg_credit_age_years",
    "bureau_total_credit", "bureau_total_debt",
    "bureau_avg_bb_dpd_rate", "bureau_total_bb_dpd_months",
    "bureau_debt_utilization", "total_debt_to_income",

    # ── INSTALLMENT ───────────────────────────────────────────────────────────
    "inst_total_payments", "inst_avg_days_late",
    "inst_pct_late", "inst_count_late", "inst_avg_underpay",

    # ── PREVIOUS APPLICATION ──────────────────────────────────────────────────
    "prev_app_count", "prev_approved_count", "prev_refused_count",
    "prev_app_rejection_rate", "prev_avg_credit",

    # ── CREDIT CARD ───────────────────────────────────────────────────────────
    "card_avg_utilization", "card_max_utilization",
    "card_avg_balance", "card_avg_payment_ratio",

    # ── POS CASH ──────────────────────────────────────────────────────────────
    "pos_count_months", "pos_pct_dpd", "pos_max_dpd", "pos_loan_count",

    # ── GOLD FEATURES (fitur lanjutan) ───────────────────────────────────────
    "combined_late_signal", "total_all_debt_to_income",
    "bureau_active_ratio", "annuity_per_person", "inst_underpay_ratio",
    "prev_credit_vs_current",
    "flag_high_card_util", "flag_high_risk_profile", "flag_no_credit_history",
    "dti_risk_bin", "credit_age_bin",
]

# Filter hanya kolom yang benar-benar ada (train punya TARGET, test tidak)
def select_existing(df, cols):
    existing = [c for c in cols if c in df.columns]
    return df.select(existing)

df_train_final = select_existing(df_train_gold, GOLD_FEATURES)
df_test_final  = select_existing(df_test_gold,  GOLD_FEATURES)

print(f"[INFO] Kolom Gold Train : {len(df_train_final.columns)}")
print(f"[INFO] Kolom Gold Test  : {len(df_test_final.columns)}")

# ==============================================================================
# BLOCK 4: SIMPAN KE GOLD
# ==============================================================================
df_train_final.repartition(20).write.mode("overwrite").parquet(GOLD_BASE + "train_features_gold/")
df_test_final.repartition(10).write.mode("overwrite").parquet(GOLD_BASE + "test_features_gold/")

print("[SUCCESS] Gold layer tersimpan di S3!")
print(f"  Train : {GOLD_BASE}train_features_gold/")
print(f"  Test  : {GOLD_BASE}test_features_gold/")

job.commit()

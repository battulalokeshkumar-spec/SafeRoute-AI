import pandas as pd
from pathlib import Path

# ==========================================
# SafeRoute AI - Data Cleaning
# ==========================================

INPUT_FILE = Path("data/accidents.csv")
OUTPUT_FILE = Path("data/cleaned_accidents.csv")

print("=" * 60)
print("          SAFEROUTE AI - DATA CLEANING")
print("=" * 60)

# Load dataset
df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal dataset shape: {df.shape}")

# ------------------------------------------
# 1. Remove duplicate records
# ------------------------------------------

duplicates = df.duplicated().sum()

print(f"\nDuplicate records found: {duplicates}")

df = df.drop_duplicates()

# ------------------------------------------
# 2. Remove unnecessary columns
# ------------------------------------------

# accident_id is only an identifier.
# festival has too many missing values.

columns_to_drop = [
    "accident_id",
    "festival"
]

df = df.drop(columns=columns_to_drop, errors="ignore")

print("\nRemoved columns:")
for column in columns_to_drop:
    print(f" - {column}")

# ------------------------------------------
# 3. Convert date and time
# ------------------------------------------

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# Convert time into a consistent format
df["time"] = pd.to_datetime(
    df["time"],
    errors="coerce"
).dt.time

# ------------------------------------------
# 4. Check missing values
# ------------------------------------------

print("\nMissing values after cleaning:")

missing = df.isnull().sum()

print(
    missing[missing > 0].to_string()
    if missing.sum() > 0
    else "No missing values found."
)

# ------------------------------------------
# 5. Remove rows with invalid dates/times
# ------------------------------------------

before = len(df)

df = df.dropna(
    subset=["date", "time"]
)

removed = before - len(df)

print(f"\nRows removed because of invalid date/time: {removed}")

# ------------------------------------------
# 6. Validate geographic coordinates
# ------------------------------------------

before = len(df)

df = df[
    df["latitude"].between(-90, 90)
    & df["longitude"].between(-180, 180)
]

removed = before - len(df)

print(f"Rows removed because of invalid coordinates: {removed}")

# ------------------------------------------
# 7. Validate risk score
# ------------------------------------------

df = df[
    df["risk_score"].between(0, 1)
]

# ------------------------------------------
# 8. Reset index
# ------------------------------------------

df = df.reset_index(drop=True)

# ------------------------------------------
# 9. Save cleaned dataset
# ------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------
# Final information
# ------------------------------------------

print("\n" + "=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)

print(f"\nOriginal rows : {len(pd.read_csv(INPUT_FILE))}")
print(f"Cleaned rows  : {len(df)}")
print(f"Columns       : {len(df.columns)}")

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nDataset is ready for ML.")
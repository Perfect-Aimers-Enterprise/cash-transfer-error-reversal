import pandas as pd

df = pd.read_csv("transactions.csv")

print("=" * 60)
print("Shape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nError Flag")
print(df["error_flag"].value_counts())

print("\nReversal Executed")
print(df["reversal_executed"].value_counts())

print("\nUnique Reversal Reasons")
print(df["reversal_reason"].value_counts())

print("\nUnique Transfer IDs")
print(df["transfer_id"].nunique())

print("\nTotal Rows")
print(len(df))

duplicates = df[df["transfer_id"].duplicated(keep=False)]

print("\nDuplicate Transfer IDs")
print(len(duplicates))

print(duplicates.head(20))

print("\nUnique Senders")
print(df["sender_id"].nunique())

print("\nUnique Beneficiaries")
print(df["beneficiary_id"].nunique())

print("\nChannels")
print(df["channel"].value_counts())

print("\nLocations")
print(df["location"].value_counts())

print("\nTop Devices")
print(df["device_id"].nunique())

print("\nTop Sessions")
print(df["session_id"].nunique())
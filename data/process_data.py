"""
source
https://www.kaggle.com/datasets/garnavaurha/shakespearify

70% of data will be used in training and 30% will be used in testing.
"""
import pandas as pd

df = pd.read_csv("/Users/jiayue/english-shakespeare-translator/data/data.csv")

# Calculate the split point
split_point = int(len(df) * 0.7)

# Split the DataFrame
data_train = df.iloc[:split_point]
data_test = df.iloc[split_point:]

# Save the DataFrames to new CSV files
data_train.to_csv("data_train.csv", index=False)
data_test.to_csv("data_test.csv", index=False)

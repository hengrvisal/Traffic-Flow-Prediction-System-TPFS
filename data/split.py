from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import os

csv_path = 'final_df_scats_october.csv'
final_df_scats_october = pd.read_csv(csv_path)

print(final_df_scats_october.head())

output_dir = 'splitted_data'

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Split the DataFrame by 'SCATS_Number'
scat_dfs = {scat: df_group for scat, df_group in final_df_scats_october.groupby('SCATS')}

# Loop over each SCATS group and save into train and test CSVs in the new directory
for scat, group_df in scat_dfs.items():
    # Split the group_df into 80% training and 20% testing without shuffling (preserving the order)
    train_df, test_df = train_test_split(group_df, test_size=0.2, shuffle=False, random_state=42)

    # Save the training set to a CSV file in the newly created directory
    train_file_path = os.path.join(output_dir, f'{scat}_train.csv')
    train_df.to_csv(train_file_path, index=False)

    # Save the testing set to a CSV file in the newly created directory
    test_file_path = os.path.join(output_dir, f'{scat}_test.csv')
    test_df.to_csv(test_file_path, index=False)

print(f"Data has been split and saved to the directory: {output_dir}")

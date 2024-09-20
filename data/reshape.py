import pandas as pd
from time_map import time_mapping

csv_path = 'ml_train_october.csv'
ml_train_october = pd.read_csv(csv_path)

# initialize the new dataframe
final_df_scats_october = pd.DataFrame()

# iterating through the entire ml_train_october DF
for i in range(len(ml_train_october)):
    # extract the relevant columns
    date = ml_train_october.iloc[i]['Date']
    cd_melways = ml_train_october.iloc[i]['CD_MELWAYS']
    flow_data = ml_train_october.iloc[i, 4:]  # extracting V00 - V95 columns
    scat_number = ml_train_october.iloc[i]['SCATS Number']
    # create a temporary dataframe for this row's data
    temp_df = pd.DataFrame({
        'Date': date,
        'CD_MELWAYS': cd_melways,
        'Scat_Number': scat_number,
        '15_Minute_Intervals': flow_data.index,
        'Flow Vehicles': flow_data.values,
    })

    # append the temporary dataframe to the final dataframe
    final_df_scats_october = pd.concat([final_df_scats_october, temp_df], ignore_index=True)

print(final_df_scats_october.head())


# Apply the mapping to the '15_Minute_Intervals' column
final_df_scats_october['15_Minute_Intervals'] = final_df_scats_october['15_Minute_Intervals'].map(time_mapping)

# For example, '1/10/06' means 1st of October, 2006
final_df_scats_october['DateTime'] = pd.to_datetime(
    final_df_scats_october['Date'] + ' ' + final_df_scats_october['15_Minute_Intervals'],
    format='%d/%m/%y %H:%M',  # Adjust the format to match your 'Date' column
    errors='coerce'
)

# Format 'DateTime' to remove seconds and keep only hours and minutes
final_df_scats_october['DateTime'] = final_df_scats_october['DateTime'].dt.strftime('%d/%m/%Y %H:%M')

# Move 'DateTime' to the first column
new_column_order = ['DateTime'] + [col for col in final_df_scats_october.columns if col != 'DateTime']

# Reorder the DataFrame columns
final_df_scats_october = final_df_scats_october[new_column_order]

# Move 'SCATS_Number' to the last position
new_column_order = [col for col in final_df_scats_october.columns if col != 'Scat_Number'] + ['Scat_Number']

# Reorder the DataFrame
final_df_scats_october = final_df_scats_october[new_column_order]

# Insert a new column '# Lane Points' in the third position (index 2) and fill it with 1
final_df_scats_october.insert(2, '# Lane Points', 1)
final_df_scats_october.insert(3, '% Observed', 100)

#Dropping CD_MELWAYS column

final_df_scats_october.drop(columns=['CD_MELWAYS', 'Date', '15_Minute_Intervals'], inplace=True)

# Step 1: Remove the 'Flow_Vehicles' column and store its data
flow_vehicles_column = final_df_scats_october.pop('Flow Vehicles')

# Step 2: Reinsert 'Flow_Vehicles' at index 1 (second position)
final_df_scats_october.insert(1, 'Flow Vehicles', flow_vehicles_column)

final_df_scats_october.rename(columns={
    'DateTime': '5 Minutes',
    'Flow Vehicles': 'Lane 1 Flow (Veh/5 Minutes)',
    'Scat_Number': 'SCATS'
}, inplace=True)

# final_df_scats_october.to_csv('final_df_scats_october.csv', index=False)

print(final_df_scats_october.head())



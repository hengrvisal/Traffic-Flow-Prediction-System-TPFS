import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'data/splitted_data/970_test.csv'

train_df = pd.read_csv(csv_path)


print(train_df.head(10))


# Assume there is a 'DateTime' column and 'Flow_Vehicles' in the data
plt.figure(figsize=(10,6))
plt.plot(train_df['5 Minutes'], train_df['Lane 1 Flow (Veh/5 Minutes)'], label='Flow Vehicles')
plt.xlabel('5 Minutes')
plt.ylabel('Lane 1 Flow (Veh/5 Minutes)')
plt.title('Traffic Flow Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()



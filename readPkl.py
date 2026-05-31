import pandas as pd
datap = r'C:\Users\dell\Desktop\aa-bbb-2024-04-28\edit-data\Omizunagidori2018_raw_data_9B36365_lb0006_24.pkl'

with open(datap, 'rb') as f:
    data = pd.read_pickle(f)

print(data)
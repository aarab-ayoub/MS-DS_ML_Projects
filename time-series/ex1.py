import pandas as pd

file_path = "data.xls"
sheet_name = "Feuil1"  

df = pd.read_excel(file_path, sheet_name=sheet_name)

df['avg3'] = df['y'].rolling(window=3, center=True).mean()

df['avg5'] = df['y'].rolling(window=5, center=True).mean()

result = df[['t', 'année', 'trim', 'y', 'avg3', 'avg5']]


print(result)

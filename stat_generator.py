import pandas as pd

sources = [
        'random_csv_0.csv',
        'random_csv_1.csv',
        'random_csv_2.csv',
        'random_csv_3.csv',
        'random_csv_4.csv'
]

for filename in sources:
    df = pd.read_csv(filename)
    df1 = df.groupby(['a','b','c']).size().reset_index().rename(columns={0:'count'})
    print(df1)
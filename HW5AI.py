import pandas as pd
import csv


file = 'cars.csv'
statsdf = pd.read_csv(file)

print(statsdf)
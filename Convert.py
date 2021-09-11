import collections
import os
import pandas as pd
from pandas import DataFrame

txt_sentoken = "D:\\Uni\\Fci_Y4_T2\\NLP\\Assignments\\Assignment_2\\TextClassification\\txt_sentoken"

columns = ['text', 'classifier']
# a_dict = collections.defaultdict(list)

d = {}
i = 0

df = pd.DataFrame()
df = df.fillna(0) # with 0s rather than NaNs

for folder in os.listdir(txt_sentoken):
    path = os.path.join(txt_sentoken, folder)
    print("folder = ", folder)
    for file in os.listdir(path):

        if file.endswith(".txt"):

            filepath = os.path.join(path, file)
            print("filepath = ", filepath)
            file = open(filepath, mode='r')

            # read all lines at once
            d[i] = file.read()
#            d[i].append("positive")
            # close the file
            file.close()
            # d[i] = pd.read_csv(filepath, sep='delimiter')
            i += 1


df = pd.DataFrame.from_dict(d, "index")
df.to_csv (r'D:\\Uni\\Fci_Y4_T2\\NLP\\Assignments\\Assignment_2\\TextClassification\\export_dataframe.csv', index = False, header=True)
print(df.head())

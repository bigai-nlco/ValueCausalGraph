#Split value_orientation.csv into 2 files: train.csv and test.csv
#Split each value's rows into 2 part, respectively store in train.csv and test.csv
#The split ratio is 0.8:0.2
#value_orientation.csv 's head looks like:
# questionnaire	item	value	agreement	question
# a same value may have multiple rows, each row represents a question about the value
# The split is based on the value column



import pandas as pd
import numpy as np
import os
import random
from sae_lens.config import DTYPE_MAP, LOCAL_SAE_MODEL_PATH

df_valuebench = pd.read_csv(os.path.join(LOCAL_SAE_MODEL_PATH, 'value_data/value_orientation.csv'))
train_csv = os.path.join(LOCAL_SAE_MODEL_PATH, 'value_data/value_orientation_train.csv')
test_csv = os.path.join(LOCAL_SAE_MODEL_PATH, 'value_data/value_orientation_test.csv')

value_list = df_valuebench['value'].unique()
#each value's rows are split into 2 parts, respectively store in train.csv and test.csv
train_list = []
test_list = []
for value in value_list:
    rows = df_valuebench[df_valuebench['value'] == value]
    rows = rows.sample(frac=1).reset_index(drop=True)  #shuffle
    split_index = int(len(rows) * 0.7)
    train_list.append(rows.iloc[:split_index])
    test_list.append(rows.iloc[split_index:])
df_train = pd.concat(train_list)
df_test = pd.concat(test_list)
df_train.to_csv(train_csv, index=False)
df_test.to_csv(test_csv, index=False)
print('split value_orientation.csv into train.csv and test.csv successfully')



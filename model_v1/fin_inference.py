import json
import pandas as pd
import numpy as np
from scipy import stats

input_cc_path = 'result_2028.json'
#df = pd.read_csv("countries_without_medals.csv")
#nomedal = df['NOC'].tolist()

with open(input_cc_path, 'r') as file:
    cc = json.load(file)

crecord = [country for country in cc.keys()]


def model_func(X, a1, a2, a3, b):
    return a1 * X + a2 * (X**2) + a3 * (X**3) + b


a1 = -0.2400
a2 = 0.0296
a3 = -0.0001
b = 0.6255

infer = {}
predicted_lower = []
predicted_upper = []

residual_variance = 2  

for country in crecord:
    
    y_pred = model_func(cc[country], a1, a2, a3, b)
    infer[country] = y_pred


    predicted_se = np.sqrt(residual_variance)

    Atest = 0.7
    t_score = stats.t.ppf(1 - (1 - Atest) / 2, df=len(crecord) - 4) 

    lower = y_pred - t_score * predicted_se
    upper = y_pred + t_score * predicted_se


    predicted_lower.append(np.maximum(lower, 0)) 
    predicted_upper.append(np.maximum(upper, 0)) 


infer = {k: int(round(v, 2)) for k, v in infer.items()}
predicted_lower = [int(round(val, 2)) for val in predicted_lower]
predicted_upper = [int(round(val, 2)) for val in predicted_upper]


csv_file_name = '2028_fin_medals.csv'

data = [[key, round(infer[key],2), predicted_lower[i], predicted_upper[i]] for i, key in enumerate(infer.keys())]

df = pd.DataFrame(data, columns=['Country', 'Predicted Result', 'Predicted Lower', 'Predicted Upper'])

df.to_csv(csv_file_name, index=False)

print(f"CSV file '{csv_file_name}' created。")


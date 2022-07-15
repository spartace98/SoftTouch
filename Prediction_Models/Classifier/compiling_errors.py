import pandas as pd
import numpy as np
import os


result_path = "SVM/config6/tripod/size_150"
prediction_errors = []

for i in range(5):
  filename = os.path.join(result_path, "wrong_pred_idx_seed%d.csv"%i)
  data = pd.read_csv(filename, header = None)
  for val in data.values:
    prediction_errors.append(val[0])

print(prediction_errors)
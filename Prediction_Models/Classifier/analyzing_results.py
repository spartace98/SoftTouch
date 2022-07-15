import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# retrieve results
classifier = "SVM"
exp = "tripod"
config_index = 6
size = 150
result_folder = os.path.join(classifier, "config%s"%config_index, exp, "size_%s"%size)
num_seeds = 5 # test for one seed first
reasons_count = {} # counts the number of errors (true, pred)
neg_count = 0
fn_count = 0

for i in range(num_seeds):
    result_for_seed = os.path.join(result_folder, "result_seed%s.csv"%i)
    data = pd.read_csv(result_for_seed, index_col = 0)

    # retrieve the results
    indices_used_in_testing = data.index
    reasons_for_errors = data["reason"]

    reasons = {x for x in reasons_for_errors if pd.notna(x)}
    for r in reasons:
        # (x, y, {z}) 
        # x is the total number of errors in the category
        # y is the number of errors in the category predicted
        # z is the set of indices of wrong predictions
        if r not in reasons_count.keys():
            reasons_count[r] = [0, 0, set()]

    true_target = data["true"]
    pred_target = data["pred"]
    run_indices = data.index

    for j in range(len(true_target)):
        y_true = true_target.iloc[j]
        y_pred = pred_target.iloc[j]

        # negatives
        if y_true == 0:
            reason = reasons_for_errors.iloc[j]
            reasons_count[reason][0] += 1

            # analyze the false positives
            if y_pred == 1:
                # print(indices_used_in_testing[i])
                reasons_count[reason][1] += 1
                reasons_count[reason][2].add(run_indices[j])

        # y_true == 1
        else:
            neg_count += 1
            # analyze the false negatives
            if y_pred == 0:
                fn_count += 1
                print("False Negative for run", run_indices[j])

print(reasons_count)
print([neg_count, fn_count])
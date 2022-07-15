"""
Analyse the distribution of errors
"""
import numpy as np
import pandas as pd
import os

# variables that changes frequently
exp = "tripod"
config_index = 6
activate_cross_corr = True
imu_step = 7
add_duration = True
control_threshold = 24.0 # imu threshold (based on variance) --> should be 0.01
contact_threshold = 50 # force threshold (based on absolute value)

results_path = os.path.join(os.path.dirname(os.path.dirname(os. getcwd())), "Results", "config%s"%config_index, exp)
target_path = os.path.join(results_path, "results_%s_binary.csv"%exp)

target = pd.read_csv(target_path)["success"]
# target = target[:num_runs]
reasons = pd.read_csv(target_path)["reason"]
# reasons = reasons[:num_runs]
num_runs = len(target)
num_samples_range = range(num_runs, num_runs+1)
indices = np.arange(0, num_runs, 1)

values_to_drop = indices[target < 0]
new_target = target.drop(values_to_drop)
reasons = reasons.drop(values_to_drop)

# make sure filter out the non trainings
print("Total collected", len(target), "Total usable", len(new_target))
assert(len(reasons) == len(new_target))

# now find the distribution of errors
indices_of_errors = new_target == 0
error_reasons = reasons[indices_of_errors]
print("Number of errors:", len(error_reasons))

# find number of success
print("Number of sucess", len(new_target) - len(error_reasons))

# find percentage of success
success_rate = 1 - (len(error_reasons) / len(new_target))
print("Success Rate:", success_rate)

# find dist errors
error_dict = {}
for i in indices:
  if target[i] == 0:
    if reasons[i] not in error_dict:
      error_dict[reasons[i]] = 1
    else:  
      error_dict[reasons[i]] += 1

print(error_dict)
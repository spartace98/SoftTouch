import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import utils

# try more combinations (greedy algorithm), matches information criteria
# to select the best sensor combination

# variables that changes frequently
exp = "indexroll"
rand_seed = 10
config_index = 3
activate_cross_corr = False
imu_step = 7
add_duration = True
isGreedy = False
ith_comb = 23
if isGreedy:
    num_sensors_to_keep = len(utils.greedy_sensor_selector(ith_comb))
else:
    num_sensors_to_keep = 5
control_threshold = 300 # imu threshold (based on variance) --> should be 0.01
contact_threshold = 50 # force threshold (based on absolute value)
# num_runs_collected = 360
# indices_runs_collected = np.arange(0, num_runs_collected, 1)
base_address = "/Users/lichao/Documents/GitHub/FoamHandLabSummerResearch/Results/config%s/%s/%s_imu/trimmed_manipulations"%(config_index, exp, exp)
# RETRIEVE TARGET DATA
# target = pd.read_csv("results_%s_binary_config%s.csv"%(exp, config_index))["success"]
results_path = os.path.join(os.path.dirname(os.path.dirname(os. getcwd())), "Results", "config%s"%config_index, exp)
binary_path = os.path.join(results_path, "results_%s_binary.csv"%exp)
target = pd.read_csv(binary_path)["success"]
num_runs_collected = len(target)
num_runs_collected = 300
target = target[:num_runs_collected]
# values_to_drop = indices_runs_collected[target < 0]
reasons = pd.read_csv(binary_path)["reason"]
reasons = reasons[:num_runs_collected]

control_error_indices = reasons.index[reasons == "no manip"].to_list()
all_acc = []
potentially_wrong_indices = []

for i in control_error_indices:
    run_num = i + 1
    filename = os.path.join(base_address, "%s_trimmed_%s.csv"%(exp, run_num))

    data = pd.read_csv(filename, header = None)
    ax = data.iloc[:, 1:31:6]
    ay = data.iloc[:, 2:31:6]
    az = data.iloc[:, 3:31:6]
    acc = pd.concat([ax, ay, az], axis = 1)
    # clear bias
    acc = acc - acc.iloc[0, :]

    all_acc.append(acc)

    # check runs that exceeed threshold
    if (acc.abs().max() > control_threshold).any():
      potentially_wrong_indices.append(run_num)

      # plt.plot(acc);plt.show()

all_acc = pd.concat(all_acc, axis = 1)
# plt.plot(all_acc.iloc[100:150, :]);plt.show()
# plt.plot(all_acc);plt.show()

print("Potentially", len(potentially_wrong_indices), "wrong indices", potentially_wrong_indices)
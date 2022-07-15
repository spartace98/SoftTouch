from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn import svm
import numpy as np
import pandas as pd
from dataloader3 import Dataloader
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error
import utils2 as utils
from features_processing import get_features_with_velocity, get_features_with_vel_acc
import os
import copy

"""
Brief description of Regresion
- Sample n angle intervals (take average in the window), where n is the timestep
- Training Set
X_i = d' * n, where d' is the number of dimensions of sensors d + 1, where 1
        is the vector previous predicted angles
        Note that the sensor data added only correspond to the timestep window
y_i = n * 1, which refers to the target value (angle) at the next time step

- Test Set
y_i = n * 1, which refers to the target value (angle) at the next time step

Use all 150 data samples
"""

# Attributes
exp = "rockII"
metric = "rotation_split"
train_type = "Ridge"
config_num = 6
number_of_timesteps = 30 # 5 steps every 1 second
threshold = 20 # 30 for translation, 20 for rotation
model = "Ridge"
normalize = True # with respect to the first success
contact_threshold = 10
num_sensors = 5
num_dims_per_sensor = 6
fs = 90
train_test_ratio = 0.8
add_duration = True # useless given random spikes
filter_check = False

# file paths
results_path = os.path.join(os.path.dirname(os.path.dirname(os. getcwd())), "Results", "config%s"%config_num, exp)
binary_path = os.path.join(results_path, "results_%s_binary.csv"%exp)
data = pd.read_csv(binary_path)
num_runs = len(data)
# num_runs = 350
success_failure_target = data["regression"][:num_runs].astype(int)
original_indices = np.arange(num_runs)
success_target_indices = original_indices[success_failure_target == 1]

# FEATURES
imu_path = os.path.join(results_path, "%s_imu"%exp, "trimmed_manipulations")
video_path = os.path.join(results_path, "%s_video"%exp, "trimmed_angles")
number_of_successful_features = len(success_target_indices)
video_step = int(90/number_of_timesteps)
imu_step = int(210/number_of_timesteps)

# IMU DATA
num_sensors_to_keep = 5
control_threshold = 300
isGreedy = False
ith_comb = None
run_indices = success_target_indices
# imu_data shape is (n, n_dim, timestep)
scaling_factor, imu_data, train_data_non_avg, control_errors_relative_index = utils.get_training_data(exp, imu_path, 
                                                                            run_indices, num_sensors, 
                                                                            num_dims_per_sensor, fs, 
                                                                            num_sensors_to_keep, 
                                                                            control_threshold = control_threshold,
                                                                            norm = normalize, 
                                                                            imu_step = imu_step, 
                                                                            isGreedy = isGreedy, 
                                                                            ith_comb = ith_comb)
# add force sensor readings to filter out non-contacts
force_readings = utils.get_force_sensor_data(exp, imu_path, run_indices, num_sensors, 1, fs, num_dims_to_keep = 5)
contact_timesteps = utils.convert_force_to_timesteps(force_readings, imu_step, contact_threshold, num_force_sensors = 5)

# include force sensor readings (binary)
if add_duration:
    # shape (n, timesteps, num_dims)
    train_data_2 = np.append(imu_data, contact_timesteps, axis = 2)
else:
    train_data_2 = imu_data

# sys.exit()

# TARGET
video_path = os.path.join(results_path, "%s_video"%exp)
apriltag_data = np.zeros((num_runs, number_of_timesteps, 1))
temp_indices = []
for index in success_target_indices:
    run = index + 1
    # print(run)
    # data_path = os.path.join(video_path, "apriltag", "apriltag_rotation_%s.csv"%(run))
    data_path = os.path.join(video_path, "apriltag", "%s_apriltag_%s_%s.csv"%(exp, metric, run))
    data_video = pd.read_csv(data_path, header = None)

    # low pass filter the apriltag rotation
    # data_video = utils.butter_lowpass_filter(data_video, cutoff = 5, fs = 30, order = 2)
    # data_video = pd.DataFrame(data_video)

    data = data_video.groupby(data_video.index // video_step).mean()

    # correct for bias
    data = data - data[0][0]
    
    # at this point, the size data is n x 36, where n is number of timesteps
    if (data.mean() < 0).all():
        data = -data
    apriltag_data[index, :, :] = data

    if data.iloc[25, 0] > 80:
        # temp_indices.append(index+1)
        # plt.plot(data)
        print(index + 1)

# plt.legend(temp_indices)
# plt.show()

# Now select only data where the maximum is +-threshold from the mean
# look at mean of the max values
max_vals = []
indices_within_range = []
for i in success_target_indices:
    max_val = max(apriltag_data[i, :, :])
    max_vals.append(max_val)
# mean_of_max_vals = np.mean(max_vals)
median_of_max_vals = np.median(max_vals)
print("Length before filter is", len(apriltag_data[success_target_indices]))
# print("Mean is", mean_of_max_vals)
print("Median is", median_of_max_vals)

# sys.exit()

for i in success_target_indices:
    max_val = max(apriltag_data[i, :, :])
    if abs(max_val - median_of_max_vals) <= threshold:
        indices_within_range.append(i)

filtered_success_target_indices = []
for i in success_target_indices:
    if i in indices_within_range:
        filtered_success_target_indices.append(i)
success_target_indices = np.array(filtered_success_target_indices)
print("Length after filter is", len(success_target_indices))

# plt.plot(apriltag_data[success_target_indices].squeeze().transpose());plt.show()
# sys.exit()

number_of_successful_features = len(success_target_indices)

restricted_size = number_of_successful_features
np.random.seed(10)
indices = np.arange(number_of_successful_features)
random_train_test_indices = np.random.choice(indices, size = restricted_size, replace = False)
chosen_train_test_indices = success_target_indices[random_train_test_indices]
root_mse_across_timesteps = []
train_data_2 = train_data_2[random_train_test_indices] # selecting only the successful ones
apriltag_data = apriltag_data[success_target_indices][random_train_test_indices]

if filter_check:
    plt.plot(apriltag_data.squeeze().transpose());plt.show()
    sys.exit()

# Train Test Split
train_size = int(train_test_ratio*restricted_size)
test_size = restricted_size - train_size
indices = np.arange(restricted_size)
random_train_indices = np.random.choice(indices, size = train_size, replace = False)
chosen_train_indices = chosen_train_test_indices[random_train_indices]
random_test_indices = np.setdiff1d(indices, random_train_indices)
chosen_test_indices = chosen_train_test_indices[random_test_indices]
assert(any(np.isin(random_train_indices, random_test_indices)) == False)
# sys.exit()

# plt.plot(apriltag_data[random_train_indices].squeeze().transpose());plt.show()
# sys.exit()

# example_predictions = []
# actual_angles = []
# previous_angle_estimates # vector
predicted_results = pd.DataFrame()
actual_angles = pd.DataFrame()
y_train_predicted_results = pd.DataFrame()
y_train_actual_angles = pd.DataFrame()

max_dist_translated = apriltag_data.squeeze().transpose().max(axis = 0)
# max_dist_translated = max(max_dist_translated[random_test_indices])
max_dist_translated = max_dist_translated.reshape((1, -1))
mean_dist_translated = apriltag_data.squeeze().transpose().mean()

previous_angles = apriltag_data[:, 0]
# original_apriltag_data = np.copy(apriltag_data)
# apriltag_data = apriltag_data.squeeze()
# sys.exit()

for i in range(number_of_timesteps-1):
    current_timestep = i + 1

    # 1. Prepare features set
    # get imu data for current timestep
    # train_data_2 = utils.get_imu_features(train_data_2, current_timestep)
    # imu_features = train_data_2[:, :current_timestep, :]
    imu_features = train_data_2[:, current_timestep, :]
    # imu_features = train_data_2[:, max(0, current_timestep-3):current_timestep, :]
    # imu_features = get_features_with_velocity(train_data_2, current_timestep, 1)
    # imu_features = get_features_with_vel_acc(train_data_2, current_timestep, 1)

    # get camera data for current timestep
    # camera = dl.get_camera_data(train_data_2, current_timestep)

    # imu_features
    
    X = np.reshape(imu_features, (restricted_size, -1)) # collapse dimensions for appending previous angles
    X = np.concatenate((X, previous_angles), axis = 1)
    # X = np.squeeze(X, axis = 1) # collapse dimensions for training

    # 2. Prepare target set
    # y = dl.get_target(current_timestep)
    # print(np.shape(apriltag_data))
    y = apriltag_data[:, current_timestep, :]

    # plt.plot(apriltag_data.squeeze().transpose());plt.show()
    # plt.scatter([current_timestep]*len(y), y)

    # do train test split here
    X_train = X[random_train_indices]
    y_train = y[random_train_indices]
    X_test = X[random_test_indices]
    y_test = y[random_test_indices]

    # Fit to regression model
    # reg = Ridge()
    # reg = LinearRegression(fit_intercept = True)
    reg = svm.SVR(kernel = "linear")
    reg.fit(X_train, y_train.squeeze())

    y_train_predicted = reg.predict(X_train)
    y_test_predicted = reg.predict(X_test.squeeze())

    # append predictions of X_train
    # previous_angles[random_train_indices] = y_train_predicted
    previous_angles[random_train_indices] = y_train
    previous_angles[random_test_indices] = y_test_predicted[:, None]
    # print(previous_angles)
    # sys.exit()

    # plt.scatter([current_timestep]*len(y_train), y_train)
    # plt.scatter([current_timestep+1]*len(y_train), y_train_predicted)
    # plt.show()

    # example_predictions.append(y_test_predicted[0])
    # actual_angles.append(y_test[0])

    y_train_predicted_results[i+1] = y_train_predicted.reshape(-1)
    y_train_actual_angles[i+1] = y_test.reshape(-1)

    predicted_results[i+1] = y_test_predicted.reshape(-1)
    actual_angles[i+1] = y_test.reshape(-1)

    # find normalized MSE (wrt to the maximum distance translated during manip)
    root_mse_error = mean_squared_error(y_test, y_test_predicted)**0.5
    # print("Root MSE Error for timestep %s is %s"%(current_timestep, root_mse_error))
    root_mse_across_timesteps.append(root_mse_error)

    # max_dist_translated = max(y_test) - min(y_test)
    # y_test = np.divide(y_test, max_dist_translated)
    # y_test_predicted = np.divide(y_test_predicted, max_dist_translated)
    # root_mse_error_normalised = mean_squared_error(y_test, y_test_predicted)**0.5
    # root_mse_error_normalised = reg.score(X_test, y_test)
    # root_mse_error_normalised = root_mse_error / mean_dist_translated

    # max_dist_translated = max(y_test) - min(y_test)
    # max_dist_translated = np.mean(y_test)
    # root_mse_error_normalised = 1 - root_mse_error/abs(np.mean((y_test)))
    # root_mse_error_normalised = mean_squared_error(y_test, y_test_predicted)**0.5 / max_dist_translated

    # print("Normalised Root MSE Error for timestep %s is %s"%(current_timestep, root_mse_error_normalised))
    # root_mse_across_timesteps.append(root_mse_error_normalised)

# sys.exit()
# plt.plot(apriltag_data.squeeze().transpose());plt.show()

total_mse = sum(root_mse_across_timesteps)
mse_across_timesteps = np.array(root_mse_across_timesteps)

# save all results in folder
result_folder = "%s_%s_config%s"%(exp, train_type, config_num)
if os.path.exists(result_folder) == False:
    os.mkdir(result_folder)

# save the mse_across_timesteps
filename = "mse_across_timesteps.csv"
np.savetxt(os.path.join(result_folder, filename), mse_across_timesteps, delimiter = ",")
print("Total MSE =", total_mse)
# print("Reg Score =", 1-total_mse/(number_of_timesteps-1))
print("Reg Score =", total_mse/(number_of_timesteps-1))
# print(total_mse/29)

filename = "y_test_predicted_values.csv"
predicted_results.index = chosen_test_indices
predicted_results.to_csv(os.path.join(result_folder, filename))

filename = "y_test_actual_values.csv"
actual_angles.index = chosen_test_indices
actual_angles.to_csv(os.path.join(result_folder, filename))

filename = "y_train_predicted_values.csv"
y_train_predicted_results.index = chosen_train_indices
y_train_predicted_results.to_csv(os.path.join(result_folder, filename))

filename = "y_train_actual_values.csv"
y_train_predicted_results.index = chosen_train_indices
y_train_actual_angles.to_csv(os.path.join(result_folder, filename))
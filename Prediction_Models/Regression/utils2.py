# Current Utilities
# 1. Sensor Selection
# 2. Data Contruction from sensor selection
from cmath import nan
from numpy.core.fromnumeric import reshape
import pandas as pd
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt
import sys
from scipy.signal import correlate
import copy

# chooses the dimensions instead of sensor
def sensor_selector2(manip_template, num_sensors, num_dims, fs):
  
    # storing entropy for each of the sensor
    entropies = np.zeros(num_sensors*num_dims)
    total_dims = num_sensors*num_dims

    for dim_index in range(total_dims):
        data = manip_template[:, dim_index]
        # normalize the data first
        # data = (data - np.mean(data)) / np.std(data)

        f, Pxx_den = signal.welch(data, fs, nperseg=128)

        # assuming white noise
        total_non_dc_power = sum(Pxx_den[1:])
        entropies[dim_index] = total_non_dc_power

    entropies_temp = []
    for i in range(num_sensors):
        start_index = i*num_dims + 3
        end_index = i*num_dims + 6
        segment = entropies[start_index:end_index]
        entropies_temp.extend(segment) 
    
    return np.argsort(entropies)[::-1], entropies

def greedy_sensor_selector(i_th_comb):
    # permutations of all sensor combinations
    # for 5 sensors, the combs are as follows
    combs = [[0], [1], [2], [3], [4],
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [1, 3], [1, 4],
            [2, 3], [2, 4],
            [3, 4], 
            [0, 1, 2], [0, 1, 3], [0, 1, 4],
            [0, 2, 3], [0, 2, 4],
            [0, 3, 4],
            [1, 2, 3], [1, 2, 4],
            [1, 3, 4],
            [2, 3, 4],
            [0, 1, 2, 3], [0, 1, 2, 4],
            [0, 2, 3, 4],
            [1, 2, 3, 4],
            [0, 1, 2, 3, 4]]

    return combs[i_th_comb]

# manip_template: single run
# chooses imu sensor
def sensor_selector(manip_template, num_sensors, num_dims, fs):
    
    # storing entropy for each of the sensor
    entropies = np.zeros(num_sensors)

    for sensor_index in range(num_sensors):
        i_min = sensor_index*num_dims
        ax = manip_template[:, i_min]
        ay = manip_template[:, i_min+1]
        az = manip_template[:, i_min+2]
        gx = manip_template[:, i_min+3]
        gy = manip_template[:, i_min+4]
        gz = manip_template[:, i_min+5]

        acc = (ax**2 + ay**2 + az**2)**0.5
        gyro = (gx**2 + gy**2 + gz**2)**0.5

        f, Pxx_den_acc = signal.welch(acc, fs, nperseg=128)
        f, Pxx_den_gyro = signal.welch(gyro, fs, nperseg=128)

        # assuming white noise
        total_non_dc_power = sum(Pxx_den_acc[1:])
        entropies[sensor_index] = total_non_dc_power

    return np.argsort(entropies)[::-1], entropies

def force_sensor_selector(manip_template, num_sensors, num_dims, fs):
    # storing entropy for each of the sensor
    entropies = np.zeros(num_sensors)

    for sensor_index in range(num_sensors):
        i_min = sensor_index*num_dims
        force_readings = manip_template[:, i_min]

        f, Pxx_den_force = signal.welch(force_readings, fs, nperseg=128)

        total_non_dc_power = sum(Pxx_den_force[1:])
        entropies[sensor_index] = total_non_dc_power

    return np.argsort(entropies)[::-1], entropies

def sensor_array_construction(base_address, start_dim_index, end_dim_index, run_indices, exp, num_samples):

    num_dims = end_dim_index - start_dim_index
    # print("num_dims", num_dims)
    # print(run_indices)
    num_runs = len(run_indices)
    unflattened_training_data = np.zeros((num_runs, num_samples, num_dims))

    # print(211 in run_indices)
    # print(212 in run_indices)
    
    for j, i in enumerate(run_indices):
        path = os.path.join(base_address, "%s_trimmed_%d.csv"%(exp, i+1))
        
        data = pd.read_csv(path, header = None)
        restricted_data = data.iloc[:, start_dim_index:end_dim_index]

        # check if there are any nan data
        anyNan = (np.nan == np.sum(restricted_data.sum()))
        if anyNan:
            print("nan detected in run", i)

            # find where the nan is
            num_samples, num_dims = np.shape(restricted_data)
            print(num_samples, num_dims)
            for i in range(num_samples):
                for j in range(num_dims):
                    # print(restricted_data.iloc[i, j])
                    if restricted_data.iloc[i, j] == np.nan:
                        print("nan detected in dim", j)
                        plt.plot(restricted_data[:, j])
                        plt.show()
            sys.exit()

        unflattened_training_data[j, :, :] = restricted_data

    return unflattened_training_data

# select chosen dimensions
def sensor_array_with_priorities2(sensor_array, dims_to_keep, num_dims = 6):

    num_runs, num_samples, num_dims_total = np.shape(sensor_array)
    sensor_array_prioritised = np.zeros((num_runs, num_samples, len(dims_to_keep)))

    sensor_array_prioritised = sensor_array[:, :, dims_to_keep]

    return sensor_array_prioritised

# select chosen sensor and take all dims
def sensor_array_with_priorities(sensor_array, sensors_to_keep, num_dims = 6):

    num_runs, num_samples, num_dims_total = np.shape(sensor_array)
    sensor_array_prioritised = np.zeros((num_runs, num_samples, len(sensors_to_keep)*num_dims))

    # check if there are any nan data
    if np.isnan(np.sum(sensor_array)):
        num_runs, num_samples, total_num_dims = np.shape(sensor_array)
        for i in range(num_runs):
            for j in range(num_samples):
                for k in range(total_num_dims):
                    if np.isnan(sensor_array[i, j, k]):
                        # fix the nan by using previous datapoint
                        sensor_array[i, j, k] = sensor_array[i, j-1, k]

    for i, sensor_index in enumerate(sensors_to_keep):
        # print("Sensor Index", sensor_index)
        i_min = sensor_index*num_dims
        i_start = i*num_dims
        sensor_array_prioritised[:, :, i_start:i_start+num_dims] = sensor_array[:, :, i_min:i_min+num_dims]

    return sensor_array_prioritised

def get_training_data(exp, base_address, run_indices, num_sensors, num_dims_per_sensor, fs, num_sensors_to_keep, control_threshold, norm = False, imu_step = 1, isGreedy = False, ith_comb = None):
    control_errors_relative_index = set()
    num_runs = len(run_indices)
    unflattened_training_data = sensor_array_construction(base_address, 1, 31, run_indices, exp, 210)
    # sensor_priorities, entropies = sensor_selector2(unflattened_training_data[0, :, :], num_sensors, num_dims, fs)
    # unflattened_training_data_prioritised = sensor_array_with_priorities2(unflattened_training_data, sensor_priorities[:num_sensors_to_keep])

    # plt.plot(unflattened_training_data[0, :])
    # plt.show()

    # choose greedy algorithm
    if isGreedy:
        comb = greedy_sensor_selector(ith_comb)
        unflattened_training_data_prioritised = sensor_array_with_priorities(unflattened_training_data, np.array(comb))
        print("IMU sensors chosen:", comb)
    else:
        # chooses the first run as successful template for sensor selector
        sensor_priorities, entropies = sensor_selector(unflattened_training_data[0, :, :], num_sensors, num_dims_per_sensor, fs)
        unflattened_training_data_prioritised = sensor_array_with_priorities(unflattened_training_data, sensor_priorities[:num_sensors_to_keep])
        print("IMU sensors chosen:", sensor_priorities[:num_sensors_to_keep])

    # clear bias
    # shape of unflattened_training_data_prioritised (num_runs_selected, duration, dims)
    unflattened_training_data_prioritised = unflattened_training_data_prioritised - unflattened_training_data_prioritised[:, 0:1, :]

    # print("dim", np.shape(unflattened_training_data_prioritised[149, :, :]))
    # plt.plot(unflattened_training_data_prioritised[0, :, 0])
    # plt.show()

    # norm (SOMETHING WRONG WITH NORM)
    if norm:
        # restricted_data = normalize(unflattened_training_data_prioritised, axis = 1)
        training_data_shape = np.shape(unflattened_training_data_prioritised)
        # print(training_data_shape)

        # Use run 1 as template success
        template_run = unflattened_training_data_prioritised[0, :, :]
        template_run_min = template_run.min(axis = 0, keepdims = True)
        template_run_max = template_run.max(axis = 0, keepdims = True)
        # print(np.shape(template_run_max))
        # sys.exit()

        template_scaling_factor = template_run_max - template_run_min

        control_errors_relative_index = np.argwhere(abs(unflattened_training_data_prioritised[:, :, 0]).max(axis = 1) < control_threshold)
        unflattened_training_data_prioritised = (unflattened_training_data_prioritised - template_run_min) / template_scaling_factor

        # print(np.shape(abs(unflattened_training_data_prioritised[:, :, 0]).max(axis = 1)))
        # sys.exit()

        # make this faster
        # for i in range(training_data_shape[0]):
            # run_data = unflattened_training_data_prioritised[i, :, :]

            # # find the control error indices
            # if max(abs(run_data[:, 0])) < control_threshold:
            #     control_errors_relative_index.add(i)

            #     # plt.plot(run_data[:, 0])
            #     # plt.show()

            # norm_run_data = (run_data - template_run_min) / template_scaling_factor
            # unflattened_training_data_prioritised[i, :, :] = norm_run_data
        
        scaling_factor = template_scaling_factor[0]
        
    else:
        # subtracting bias
        scaling_factor = [1]

    unflattened_training_data_prioritised_copy = copy.deepcopy(unflattened_training_data_prioritised)
    # plt.plot(unflattened_training_data_prioritised_copy[149, :, 0])
    # plt.show()

    # condense by averaging
    if imu_step > 1:
        # cast to higher dimensions
        condensed_run = unflattened_training_data_prioritised.reshape(num_runs, -1, imu_step, num_dims_per_sensor*num_sensors_to_keep)
        unflattened_training_data_prioritised = np.mean(condensed_run, axis = 2)

    # collapse to num_runs, num_samples*num_dims
    return scaling_factor, \
            unflattened_training_data_prioritised, \
            unflattened_training_data_prioritised_copy, \
            control_errors_relative_index

def get_force_sensor_data(exp, base_address, run_indices, num_sensors, num_dims, fs, num_dims_to_keep, imu_step = 1):
    unflattened_training_data = sensor_array_construction(base_address, 31, 36, run_indices, exp, 210)
    unflattened_training_data = abs(unflattened_training_data) # convert force readings to positive
    sensor_priorities, entropies = force_sensor_selector(unflattened_training_data[0, :, :], num_sensors, num_dims, fs)
    unflattened_training_data_prioritised = sensor_array_with_priorities(unflattened_training_data, sensor_priorities[:num_dims_to_keep], num_dims)
    
    num_runs, num_samples, num_dims_prioritised = np.shape(unflattened_training_data_prioritised)
    # return np.reshape(unflattened_training_data_prioritised, (num_runs, -1), order = "F")

    # now reshape the force array
    if imu_step > 1:
        unflattened_training_data_prioritised_reshaped = unflattened_training_data_prioritised.reshape(num_runs, -1, imu_step, num_dims_to_keep)
        unflattened_training_data_prioritised_condensed = np.mean(unflattened_training_data_prioritised_reshaped, axis = 2)

        return unflattened_training_data_prioritised_condensed.squeeze()
    
    return unflattened_training_data_prioritised

# calculate proportion of time above threshold
def get_force_contact_duration(force_readings, threshold):
    num_runs, num_samples = np.shape(force_readings)
    contact_duration = np.zeros((num_runs, 1))

    for i in range(num_runs):
        num_samples_above_threshold = sum(force_readings[i] > threshold)
        proportion_above_threshold = num_samples_above_threshold / num_samples
        contact_duration[i] = proportion_above_threshold

    return contact_duration

# normalize to value between 0 and 1
def normalize(data):
    if np.isnan(np.sum(data)):
        sys.exit()
    
    elif (max(data) - min(data)) == 0:
        return np.zeros(np.shape(data))
    else:
        return (data - min(data)) / (max(data) - min(data))

# calculate proportion of time above threshold
# calculate mean of the forces
def get_force_contact_duration2(force_readings, threshold):
    num_runs, num_samples, num_force_sensors = np.shape(force_readings)
    contact_duration = np.zeros((num_runs, num_force_sensors))

    for i in range(num_runs):
        # set nan to 0s
        force_readings_for_run = np.nan_to_num(force_readings[i])
        # apply low pass filter?

        # duration of contact
        # num_samples_above_threshold = np.count_nonzero(force_readings_for_run > threshold)
        num_samples_above_threshold = sum(force_readings[i] > threshold)

        proportion_above_threshold = num_samples_above_threshold / num_samples
        
        # valid_samples_above_threshold = force_readings_for_run[force_readings_for_run > threshold]

        contact_duration[i, :] = proportion_above_threshold

    return contact_duration

def get_control_contact_error_indices(train_data, force_readings, num_condensed_samples_per_dim, control_threshold, contact_threshold):
    num_runs, _ = np.shape(train_data)
    control_errors = set()
    non_contacts_errors = set()

    std_accum  = []

    for i in range(num_runs):
        # just reading ax of the highest priority sensor
        data = train_data[i, :num_condensed_samples_per_dim] - train_data[i, 0]
        segment = data

        # plt.plot(segment);plt.show()
        force = abs(force_readings[i, :num_condensed_samples_per_dim] - force_readings[i, 0])

        std_accum.append(np.std(segment))

        if np.std(segment) < control_threshold:
            control_errors.add(i)
            print("control error at index", i)
            plt.plot(segment);plt.show()

        elif np.max(force) < contact_threshold:
            non_contacts_errors.add(i)

    # plt.plot(std_accum);plt.show()
    
    return control_errors, non_contacts_errors

def get_cross_corr(train_data, num_runs, num_dims_per_sensor, num_dims_to_keep, num_condensed_samples_per_dim, num_features_with_cross_corr):
    half_num_condensed_samples_per_dim = num_condensed_samples_per_dim // 2
    result = np.zeros((num_runs, num_dims_per_sensor*num_dims_to_keep*num_features_with_cross_corr))
    print(num_dims_per_sensor, num_dims_to_keep, num_features_with_cross_corr)

    # print("Length of train data", len(train_data))
    # print("Num condensed dims", num_condensed_samples_per_dim)
    # print("Total num dims", num_dims_per_sensor*num_dims_to_keep)
    # print(np.shape(train_data))
    # sys.exit()

    # correlating first half and last half
    for i in range(len(train_data)):
        for j in range(num_dims_per_sensor*num_dims_to_keep):
            j_min = j*num_condensed_samples_per_dim
            j_max = j_min+num_condensed_samples_per_dim
            data = train_data[i, j_min:j_max]

            # normalize the data so that spikes do not cause high correlations
            data = (data - max(data)) / (max(data) - min(data))

            # split data into two and do cross correlation
            data_start = data[:half_num_condensed_samples_per_dim]

            # acceleration
            if j < 3:
                # continue
                data_end = data[half_num_condensed_samples_per_dim:]
            # gyroscope
            else:
                # continue
                data_end = -data[half_num_condensed_samples_per_dim:]

            corr_coeff = np.convolve(data_start, data_end)
            # should be max since convolution might be negative
            max_coeff = np.max(abs(corr_coeff))
            result[i, num_features_with_cross_corr*j] = max_coeff
            result[i, num_features_with_cross_corr*j+1] = np.std(data)
            result[i, num_features_with_cross_corr*j+2] = np.mean(data)
            # result[i, num_features_with_cross_corr*j+3] = np.max(data)
            # result[i, num_features_with_cross_corr*j+4] = np.min(data)
    
    # print(np.shape(result))
    # plt.plot(result[:, :num_dims_per_sensor]);plt.show()
    # sys.exit()
    return result

# analyzes the distribution of the target
def binary_analysis(target):
    index_of_zeros = np.where(target == 0)
    index_of_ones = np.where(target == 1)

    return index_of_zeros[0], index_of_ones[0]

def get_data_distribution(target):
    index_of_zeros, index_of_ones = binary_analysis(target)
    return len(index_of_ones) / (len(index_of_zeros) + len(index_of_ones))


def convert_force_to_timesteps(force_readings, timestep_interval, contact_threshold, num_force_sensors):
    # np.reshape(unflattened_training_data_prioritised, (num_runs, -1), order = "F")

    # convert to binary values for contact
    contact_readings = (force_readings > contact_threshold).astype(float)

    # now condense
    num_runs = len(contact_readings)
    force_readings_condensed = contact_readings.reshape(num_runs, -1, timestep_interval, num_force_sensors)
    force_readings_condensed = np.sum(force_readings_condensed, axis = 2)

    # now convert back to binary value
    force_readings_condensed = (force_readings_condensed > 0).astype(float)

    return force_readings_condensed
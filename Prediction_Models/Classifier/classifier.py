from pandas.core.frame import DataFrame
from sklearn import svm
import pandas as pd
import numpy as np
import sys
import os
from dataloader import Dataloader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, accuracy_score
import utils
from sampling_dataset import random_sampling

# Attributes
model = "SVM"
train_type_2 = "imu"
number_of_seeds = 5
train_test_ratio = 0.2
show_each_seed = False
show_con_mat = True
majority = None
size_of_test_set = 20
num_sensors = 5
num_dims_per_sensor = 6
fs = 90
num_features_with_cross_corr = 5
num_samples_per_dim = 210
normalize = True # with respect to the first success

# try more combinations (greedy algorithm), matches information criteria
# to select the best sensor combination

# variables that changes frequently
exp = "rockII"
rand_seed = 10
config_index = 3
activate_cross_corr = False
imu_step = 7
add_duration = True
isGreedy = False # for sensor selection
ith_comb = 23
if isGreedy:
    num_sensors_to_keep = len(utils.greedy_sensor_selector(ith_comb))
else:
    num_sensors_to_keep = 5
control_threshold = 300 # imu threshold (based on variance) --> should be 0.01
contact_threshold = 50 # force threshold (based on absolute value)
repo_path = os.path.dirname((os.path.dirname(os.getcwd())))
base_address = os.path.join(repo_path, "data", "config%s"%config_index, exp, "%s_imu"%exp)

# create result file to store train results
result_folder = os.path.join(model, "config%s"%config_index)
if os.path.exists(result_folder) == False:
    os.mkdir(result_folder)
result_folder = os.path.join(model, "config%s"%config_index, exp)
if os.path.exists(result_folder) == False:
    os.mkdir(result_folder)

# RETRIEVE TARGET DATA
# target = pd.read_csv("results_%s_binary_config%s.csv"%(exp, config_index))["success"]
results_path = os.path.join(repo_path, "data", "config%s"%config_index, exp)
binary_path = os.path.join(results_path, "results_%s_binary.csv"%exp)

target = pd.read_csv(binary_path)["success"]
num_runs_collected = len(target)
# num_runs_collected = 300
target = target[:num_runs_collected]
# values_to_drop = indices_runs_collected[target < 0]
reasons = pd.read_csv(binary_path)["reason"]
reasons = reasons[:num_runs_collected]


# RETRIEVE TRAINING DATA
# remove -1 indices before starting sampling process
run_indices_to_ignore_set = set()
run_indices_to_ignore = np.argwhere(target.values < 0)
for i in run_indices_to_ignore:
    run_indices_to_ignore_set.add(i[0])

run_indices = random_sampling(reasons, run_indices_to_ignore_set, num_success = 120, num_slip = 10, num_no_manip = 10, num_no_obj = 10, rand_seed = rand_seed)
num_runs = len(run_indices)
scaling_factor, train_data_2, train_data_non_avg, control_errors_relative_index = utils.get_training_data(exp, base_address, 
                                                                            run_indices, num_sensors, 
                                                                            num_dims_per_sensor, fs, 
                                                                            num_sensors_to_keep, 
                                                                            control_threshold = control_threshold,
                                                                            norm = normalize, 
                                                                            imu_step = imu_step, 
                                                                            isGreedy = isGreedy, 
                                                                            ith_comb = ith_comb)
num_condensed_samples_per_dim = num_samples_per_dim // imu_step
# add force sensor readings to filter out non-contacts
force_readings = utils.get_force_sensor_data(exp, base_address, run_indices, num_sensors, 1, fs, num_dims_to_keep = 5, imu_step = 1) # no condensing for est duration
contact_duration = utils.get_force_contact_duration2(force_readings, contact_threshold)
# apply thresholding at this step to filter out control errors and non contacts

control_errors = [run_indices[i[0]] for i in control_errors_relative_index]

# find the non filtered control errors
# actual_control_errors_indices = np.arange(num_runs, num_runs-10, -1)
actual_control_errors_indices = []
for i in run_indices:
    if reasons[i] == "no manip":
        actual_control_errors_indices.append(i)
# sys.exit()

# find intersection between actual_control_errors and filtered actual_control_errors
correctly_filtered_indices = np.intersect1d(actual_control_errors_indices, list(control_errors))
print("Number of control errors detected: ", len(control_errors))
print("Number of total control errors", len(actual_control_errors_indices))
print("Number of correctly detected control errors", len(correctly_filtered_indices))

# now find the non filtered control errors
misses = np.setxor1d(actual_control_errors_indices, correctly_filtered_indices)
print("Misses:", misses)

# sys.exit()

# if train on cross correlation
if activate_cross_corr:
    train_data_supplement = utils.get_cross_corr(train_data_2, num_runs, num_dims_per_sensor, 
                                        num_sensors_to_keep, num_condensed_samples_per_dim, 
                                        num_features_with_cross_corr)
    train_data_2 = np.append(train_data_2, train_data_supplement, axis = 1)

# append this new atttribute
train_type = train_type_2
if add_duration:
    train_data_2 = np.append(train_data_2, contact_duration, axis = 1)
train_data_2 = pd.DataFrame(train_data_2, index = run_indices)

target = target[run_indices]
reasons = reasons[run_indices]

for i in range(1, 2):
    train_data = train_data_2
    accuracy_across_diff_sizes = []
    auroc_across_diff_sizes = []
    natural_dist_test_set = []

    # Check if the data is trainable
    to_train = True
    # temp_target = target.drop(labels = control_errors)
    if sum(target == 1) == len(target):
        to_train = False
    elif sum(target == 0) == len(target):
        to_train = False

    # accuracy and auroc scores
    test_scores = []
    aurocs = []

    # now check if the filtered controlled errors are correct
    if to_train == False:
        seed_count = 1
        print("Untrainable, testing on control failures")

        test_score = accuracy_score(target[control_errors], np.zeros(len(control_errors)))
        test_scores.append(test_score)
        
        auroc = roc_auc_score(target[control_errors], np.zeros(len(control_errors)))
        aurocs.append(auroc)

    else:
        # num_runs_to_train = num_runs - len(values_to_drop) - len(control_errors)
        num_runs_to_train = num_runs - len(control_errors)

        run_result_path = os.path.join(result_folder, "size_%d"%num_runs)
        if os.path.exists(run_result_path) == False:
            os.mkdir(run_result_path)

        print("Sampling", num_runs_to_train, "data")
        seed_count = 0 # while loop to ensure that we only sample seeds that work
        loop_count = 0
        while seed_count < number_of_seeds:

            if loop_count > 50:
                print("CANT FIND WORKING SEED")
                sys.exit()

            control_failure_train_indices = []
            control_failure_test_indices = []
            dl = Dataloader(train_data, target, train_test_ratio, size_of_test_set, random_state = seed_count, isTest = True)
            best_hyperparam = None
            best_hyperparam_score = 0

            X_train, X_test, y_train, y_test = dl.load_train_test_data(num_samples = num_runs)

            # only train and test on runs that do not have control failures
            for index in dl.train_idx:
                if index in control_errors:
                    control_failure_train_indices.append(index)
            for index in dl.test_idx:
                if index in control_errors:
                    control_failure_test_indices.append(index)

            X_train_filtered = X_train.drop(control_failure_train_indices)
            X_test_filtered = X_test.drop(control_failure_test_indices)
            y_train_filtered = y_train.drop(control_failure_train_indices)
            y_test_filtered = y_test.drop(control_failure_test_indices)

            # asserting no overlap of training and testing indices
            test = np.intersect1d(X_train_filtered.index, X_test_filtered.index)
            assert(test.any() == False)

            # train
            clf = svm.SVC(kernel = "linear")
            # print(X_train_filtered.values)
            clf.fit(X_train_filtered.values, y_train_filtered.values)
            try:
                clf.fit(X_train_filtered.values, y_train_filtered.values)
            except:
                loop_count += 1
                continue

            # Predict on test set
            y_pred_filtered = clf.predict(X_test_filtered.values)
            y_pred_filtered = pd.Series(y_pred_filtered, index = X_test_filtered.index)
            y_pred_control_failures = np.zeros(len(y_test) - len(y_pred_filtered))
            y_pred_control_failures = pd.Series(y_pred_control_failures, index = control_failure_test_indices)

            y_pred_reconstructed = pd.concat((y_pred_filtered, y_pred_control_failures))
            y_test_control_failures = y_test[control_failure_test_indices]
            y_test_reconstructed = pd.concat((y_test_filtered, y_test_control_failures))

            # test_score = clf.score(X_test.values, y_test.values)
            test_score = accuracy_score(y_test_reconstructed, y_pred_reconstructed)
            # print("Test score is", test_score)
            test_scores.append(test_score)

            auroc_value = roc_auc_score(y_test_reconstructed, y_pred_reconstructed)
            aurocs.append(auroc_value)
            # print("auroc is", auroc_value)

            neg_class_length = sum(y_test_reconstructed == 0)
            pos_class_length = sum(y_test_reconstructed == 1)
            # print(neg_class_length, pos_class_length)
            if pos_class_length > neg_class_length:
                majority = 1
            else:
                majority = 0


            # SAVING RESULTS
            # save predicted and true
            filename = os.path.join(run_result_path, "result_seed%d.csv"%seed_count)
            indices_to_sort = np.argsort(y_test_reconstructed.index)
            y_predicted_sorted = y_pred_reconstructed.values[indices_to_sort]
            y_test_sorted = y_test_reconstructed.values[indices_to_sort]
            ordered_test_indices = y_test_reconstructed.index[indices_to_sort]
            reasons_for_indices = reasons[ordered_test_indices]

            compiled_results = np.transpose(np.vstack((ordered_test_indices+1, y_test_sorted, y_predicted_sorted)))
            compiled_results = pd.DataFrame(compiled_results, columns = ["runs", "true", "pred"])
            compiled_results["reason"] = reasons_for_indices.values
            compiled_results.to_csv(filename, index = False)

            # save train indices
            filename = os.path.join(run_result_path, "train_idx_seed%d.csv"%seed_count)
            np.savetxt(filename, (y_train.index + 1).astype(int), fmt='%i', delimiter = ",")

            # save test indices
            filename = os.path.join(run_result_path, "test_idx_seed%d.csv"%seed_count)
            np.savetxt(filename, (y_test_reconstructed.index + 1).astype(int), fmt='%i', delimiter = ",")

            # find out the indices that have been predicted incorrectly
            indices = dl.get_wrong_pred_runs(y_pred_reconstructed, y_test_reconstructed)
            if indices.size != 0:
                filename = os.path.join(run_result_path, "wrong_pred_idx_seed%d.csv"%seed_count)
                np.savetxt(filename, indices.astype(int), fmt='%i', delimiter = ",")
            
            # Confusion Matrix
            if show_con_mat:
                con_mat = confusion_matrix(y_test_reconstructed, y_pred_reconstructed)
                if len(con_mat) == 1:
                    singular_value = con_mat[0][0]
                    if majority == 0:
                        print("WRONG!")
                        con_mat = np.array([[singular_value, 0], [0, 0]])
                    else:
                        con_mat = np.array([[0, 0], [0, singular_value]])

                disp = ConfusionMatrixDisplay(confusion_matrix = con_mat, display_labels = ["failure", "success"])
                disp.plot(cmap = "Blues")

                # save conmat in result file
                filename = os.path.join(run_result_path, "confusion_matrix_seed%d.png"%seed_count)
                plt.savefig(filename)
                plt.close()

            if show_each_seed:
                print(np.array(y_test_reconstructed))
                print(y_pred_reconstructed)
                print(np.array(y_test_reconstructed) == y_pred_reconstructed)
                print("")

            seed_count += 1
            loop_count += 1

    # save train and test results
    avg_score_across_test_seeds = sum(test_scores) / seed_count
    print("Mean Accuracy", avg_score_across_test_seeds)

    avg_auroc_across_test_seeds = sum(aurocs) / seed_count
    print("AUROC Score", avg_auroc_across_test_seeds)

    if to_train:
        dist = dl.get_data_distribution()
    else:
        dist = utils.get_data_distribution(target[control_errors])
    print("Natural Dist is", dist)
    natural_dist_test_set.append(dist)

    accuracy_across_diff_sizes.append(avg_score_across_test_seeds)
    auroc_across_diff_sizes.append(avg_auroc_across_test_seeds)
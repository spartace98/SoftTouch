# script to sample dataset in a specific way

# total runs to train: 200
# success: 165
# failures: no manip (20), slip (10), no object (5)

import numpy as np
import random
import math

def random_sampling(reasons, run_indices_to_ignore, num_success, num_slip, num_no_manip, num_no_obj, rand_seed):
    indices_for_success = []
    indices_for_slip = []
    indices_for_no_manip = []
    indices_for_no_obj = []
    
    # collect the indices for the reasons
    for i in range(len(reasons)):
        if i in run_indices_to_ignore:
            continue
        if reasons[i] == "slip":
            indices_for_slip.append(i)
        elif reasons[i] == "no manip":
            indices_for_no_manip.append(i)
        elif reasons[i] == "no pen" or reasons[i] == "no object" or reasons[i] == "no marble":
            indices_for_no_obj.append(i)
        else:
            indices_for_success.append(i)

    # set the random seed
    random.seed(rand_seed)

    # randomly select the indices based on defined numbers
    print("Number of success manip", len(indices_for_success))
    print("Number of slips", len(indices_for_slip))
    print("Number of no manip", len(indices_for_no_manip))
    print("Number of no object", len(indices_for_no_obj))
    rand_indices_for_success = random.sample(indices_for_success, num_success)
    rand_indices_for_slip = random.sample(indices_for_slip, num_slip)
    rand_indices_for_no_manip = random.sample(indices_for_no_manip, num_no_manip)
    rand_indices_for_no_obj = random.sample(indices_for_no_obj, num_no_obj)

    # print(len(rand_indices_for_success), len(rand_indices_for_slip), len(rand_indices_for_no_manip), len(rand_indices_for_no_obj))
    # print(len(rand_indices_for_success + rand_indices_for_slip + rand_indices_for_no_obj + rand_indices_for_no_manip))

    # concatenating these indices
    return rand_indices_for_success + rand_indices_for_slip + rand_indices_for_no_obj + rand_indices_for_no_manip
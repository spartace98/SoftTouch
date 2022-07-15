import os

curr_dir = os.getcwd()
files_in_curr_dir = os.listdir()

# make the folders first
foldernames = ["acc_0", "acc_1", "acc_2", "acc_3", "acc_4",
                "gyro_0", "gyro_1", "gyro_2", "gyro_3", "gyro_4",
                "forces"]

for foldername in foldernames:
    if os.path.exists(foldername):
        continue
    else:
        os.mkdir(foldername)

for file in files_in_curr_dir:
    # skip the foldernames
    if os.path.isdir(file):
        continue

    for foldername in foldernames:
        if file.startswith(foldername):
            new_filename = os.path.join(foldername, file)
            os.rename(file, new_filename)


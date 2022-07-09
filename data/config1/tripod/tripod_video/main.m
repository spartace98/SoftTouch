clear all;
close all;
clc;

num_sensors = 6;
num_dims = 1;
total_dims = num_dims*num_sensors; % 6 sensors, 2 dims each
samples_per_manip = int32(20);
sampling_range = 75;
result = zeros(40, samples_per_manip*total_dims);
peaks = -1;
cutoff_freq = 5;
sample_rate = 30;

% Acc and Gyro calib
calib_filename = fullfile("trimmed_manipulations", sprintf("tripod_trimmed_%d.csv", 1));
calib_data = readtable(calib_filename);
calib_data = calib_data{:, :};
calib_readings = calib_data(:, 2:end);

calib_readings = calib_readings - calib_readings(1, :);
calib_readings = condense_dimensions(calib_readings);
calib_mean_1 = mean(calib_readings);
calib_std_1 = std(calib_readings);

calib_filename = fullfile("trimmed_manipulations", sprintf("tripod_trimmed_%d.csv", 2));
calib_data = readtable(calib_filename);
calib_data = calib_data{:, :};
calib_readings = calib_data(:, 2:end);

calib_readings = calib_readings - calib_readings(1, :);
calib_readings = condense_dimensions(calib_readings);
calib_mean_2 = mean(calib_readings);
calib_std_2 = std(calib_readings);

for i=1:80
    % Retrieve data
    filename = fullfile("trimmed_manipulations", sprintf("tripod_trimmed_%d.csv", i));
    data = readtable(filename);
    data = data{:, :};
    times = data(:, 1);
    readings = data(:, 2:end);

    if (rem(i, 2) == 1)
        calib_mean = calib_mean_1;
        calib_std = calib_std_1;
    else
        calib_mean = calib_mean_2;
        calib_std = calib_std_2;       
    end
    
    % always pick samples within 200 samples from peak(1) and peak (2)
    result(i, :) = get_samples(readings, samples_per_manip, cutoff_freq, sample_rate, calib_mean, calib_std);
    
    logger = sprintf("Completed manip %d", i);
    disp(logger);
   
end
% writematrix(result, "training_data_tripod_cam.csv");
% writematrix(result, "training_data_tripod_cam_1.csv");
% writematrix(result, "training_data_tripod_cam_2.csv");

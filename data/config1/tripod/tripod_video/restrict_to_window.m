clear all;
close all;
clc;

num_sensors = 6;
num_dims = 2;
total_dims = num_dims*num_sensors; % 6 sensors, 2 dims each
samples_per_manip = int32(20);
sampling_range = 75;
result = zeros(40, samples_per_manip*total_dims);
result_folder = "split_manipulations";
peaks = -1;
cutoff_freq = 1;
sample_rate = 30;

for i=1:180
    % Retrieve data
    filename = fullfile("split_manipulations", sprintf("tripod_formatted_%d.csv", i));
    data = readtable(filename);
    data = data{:, :};
    times = data(:, 1);
    readings = data(:, 2:end);
    sample_rate = length(times) / (times(end) - times(1));
    
    % first identify the peaks
    cx1 = readings(:, 11);
    cy1 = readings(:, 12);
    if (peaks == -1) % have not detected peaks
        pos_combined = (cx1.^2 + cy1.^2).^0.5;
        grad = gradient(pos_combined);
        pos_combined_grad = grad - mean(grad(1:100));
        pos_combined_grad = lowpass(pos_combined_grad, 1, sample_rate);
        pos_combined_grad = pos_combined_grad - pos_combined_grad(1);
        idx = find(abs(pos_combined_grad) > 1.5);
        
        if (isempty(idx))
            peaks = [1 1 1 1];
            disp("EMPTY!");
        else
            peaks = get_window_intervals(idx);
        end
    end

    % always pick samples within sampling_range samples from peak(1) and peak(3)
    i0 = peaks(1);
    i1 = i0 + sampling_range-1;
    result = data(i0:i1, :);
    
    % save the trimmed samples
    result_filename = fullfile("trimmed_manipulations", sprintf("tripod_trimmed_%d.csv", i));
    writematrix(result, result_filename);
    
%     figure();
%     subplot(2, 1, 1);
%     plot(data(:, 2));
%     subplot(2, 1, 2);
%     plot(i0:i1, data(i0:i1, 2));
        
    logger = sprintf("Completed manip %d", i);
    disp(logger);
end
% writematrix(result, "training_data_tripod_imu.csv");
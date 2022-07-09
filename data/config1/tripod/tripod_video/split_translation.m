clear all;
close all;
clc;
% errors: 79(0), 80(0), 81(0), 87(first manip)

results = zeros(90, 1);
sample_rate = 30;
cutoff_freq = 1;
offset = 5;
for i=101:109
    total_dim = 1;
    result_folder = "apriltag";
    runfile = sprintf("apriltag_raw/april_tag_vtranslation_%d_part1.csv", i);

    data = table2array(readtable(runfile));
    data = data(5:end, 2) - data(5, 2);
    % account for broken file
    if (isempty(data))
        results(2*i-1) = 0;
        results(2*i) = 0;
        continue;
    end
    
    data = lowpass(data, cutoff_freq, sample_rate);
    
    % find gradient
    grads = gradient(data);
    grads = grads(offset:end);
    idx = find(grads > 2);
    idx = idx + offset - 3;
    if (isempty(idx))
        results(2*i-1) = 0;
        results(2*i) = 0;
    else
        peaks = get_window_intervals(idx);

        length_of_data = length(data);
        first_manip_min_idx = max(1, peaks(1));
%         first_manip_min_idx = 73;
        first_manip_max_idx = min(first_manip_min_idx + 79, length_of_data);
        first_manip_interval = data(first_manip_min_idx:first_manip_max_idx);
        first_manip_interval = first_manip_interval - first_manip_interval(1);
        
        second_manip_min_idx = max(1, peaks(2));
%         second_manip_min_idx = 73;
        second_manip_max_idx = min(second_manip_min_idx + 79, length_of_data);
        second_manip_interval = data(second_manip_min_idx:second_manip_max_idx);
        second_manip_interval = second_manip_interval - second_manip_interval(1);
        
        % resample
        first_manip_interval = resample(first_manip_interval, 90, 80);
        second_manip_interval = resample(second_manip_interval, 90, 80);
        
        new_filename_first = fullfile(result_folder, sprintf("tripod_apriltag_translation_split_%d.csv", ((2*i)-1)));
        new_filename_second = fullfile(result_folder, sprintf("tripod_apriltag_translation_split_%d.csv", (2*i)));
        writematrix(first_manip_interval, new_filename_first);
        writematrix(second_manip_interval, new_filename_second); 

%         subplot(3, 1, 1);
%         plot(data);
%         subplot(3, 1, 2);
%         plot(first_manip_interval);
%         subplot(3, 1, 3);
%         plot(second_manip_interval);
    end
    
    logger = sprintf("Completed manip %d", i);
    disp(logger);
end

% results_filename = "april_tag_target_angle.csv";
% writematrix(results, results_filename);
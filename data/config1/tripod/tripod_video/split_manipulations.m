clear all;
close all;
clc;

total_dims = 2*5; % 5 sensors, 2 dims each
% samples_per_manip = 10;
% result = zeros(40, 30*samples_per_manip);
result_folder = "split_manipulations";
for i=79:79
    % Retrieve data
    filename = sprintf("spots_loc_%d.csv", i);
    data = readtable(filename);
    cutoff_freq = 1;
    sample_rate = 30;
    
    % loop through only first sensor
    peaks = -1;
    % thumb
    cx1 = table2array(data(:, 8));
    cy1 = table2array(data(:, 9));
    cx2 = table2array(data(:, 10));
    cy2 = table2array(data(:, 11));
    cx3 = table2array(data(:, 12));
    cy3 = table2array(data(:, 13));

    % first identify the peaks
    if (peaks == -1) % have not detected peaks
%         cx_offset = mean(cx3(1:100));
%         cy_offset = mean(cy3(1:100));
%         cx3 = lowpass(cx3 - cx_offset, 0.5, sample_rate) + cx_offset;
%         cy3 = lowpass(cy3 - cy_offset, 0.5, sample_rate) + cx_offset;
        
        pos_combined = (cx3.^2 + cy3.^2).^0.5;
%         pos_combined = lowpass(pos_combined - mean(pos_combined(1:100)), 1, sample_rate);
        
%         plot(pos_combined);
        
        grad = gradient(pos_combined);
        pos_combined_grad = grad - mean(grad(1:100));
        
%         figure();
%         plot(pos_combined_grad);
        
        pos_combined_grad = lowpass(pos_combined_grad, 1, sample_rate);
%         pos_combined_grad = pos_combined_grad(80:end);
%         figure();
%         plot(pos_combined_grad);
        
        idx = find(abs(pos_combined_grad) > 0.5);
        peaks = get_window_intervals(idx);
%         peaks = peaks + 80;
        
    end

    % then split each file into two subsections for each manipulation
    first_peak = peaks(1);
%     first_peak = 60;
    second_peak = peaks(3);
    
    % save 150 samples before first peak
%     first_manip = data(first_peak-30:first_peak+90, :);
    first_manip = data(first_peak:first_peak+120, :);
    
    % save 150 samples before second peak
    second_manip = data(second_peak-30:second_peak+90, :);
%     second_manip = data(second_peak:second_peak+120, :);
    
    plot(table2array(first_manip(:, 13)));
    figure();
    plot(table2array(second_manip(:, 13)));
%     break;
    
    new_filename_first = fullfile(result_folder, sprintf("tripod_formatted_%d.csv", ((2*i)-1)));
    new_filename_second = fullfile(result_folder, sprintf("tripod_formatted_%d.csv", (2*i)));
	writetable(first_manip, new_filename_first);
    writetable(second_manip, new_filename_second); 
end
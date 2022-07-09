function [pos_calib] = get_calib_samples(calib_readings, i0, i1, samples_per_manip, cutoff_freq, sample_rate)
%GET_CALIB_SAMPLES Summary of this function goes here
%   Detailed explanation goes here
    
    % Correct for Bias
    pos_calib = calib_readings - mean(calib_readings(i0, :), 1);

    % Low Pass Filter
    pos_calib = lowpass(pos_calib, cutoff_freq, sample_rate);
%     plot(pos_calib);

    % Correct for Bias after Low Pass
    pos_calib = pos_calib - mean(pos_calib(i0, :), 1);
    pos_calib = pos_calib(i0:i1, :);
end


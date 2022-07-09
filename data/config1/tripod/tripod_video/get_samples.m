function [res] = get_samples(trimmed_data, samples_per_manip, cutoff_freq, sample_rate, calib_mean, calib_std)
%GET_SAMPLES Summary of this function goes here
%   Detailed explanation goes here

    interval = length(trimmed_data) / int32(samples_per_manip);
    half_interval = idivide(interval, int32(2));
    
    % Correct for Bias
    trimmed_data = trimmed_data - trimmed_data(1, :);

    % Condense dimensions
    condensed_readings = condense_dimensions(trimmed_data);
    
    % Padding
    length_of_padding = samples_per_manip - length(condensed_readings);
    padding = zeros(length_of_padding, 6);
    trimmed_pad = vertcat(condensed_readings, padding);
    
    % Standardize to Calibration
    trimmed_pad_stand = (trimmed_pad - calib_mean) ./ std(calib_std);
    
    % Correct for Bias
    trimmed_pad_stand = trimmed_pad_stand - trimmed_pad_stand(1, :);
    
    % Low Pass Filter
    trimmed_pad_stand_fil = lowpass(trimmed_pad_stand, cutoff_freq, sample_rate);

    % Correct for Bias after Low Pass
%     trimmed_stand_fil = trimmed_stand_fil - trimmed_pad_stand_fil(1, :);
    
    trimmed_pad_fil_stand_samples = trimmed_pad_stand_fil(1+half_interval:interval:end, :);
%     trimmed_stand_fil_samples = trimmed_stand_fil_samples(1:samples_per_manip, :);

    if (height(trimmed_pad_fil_stand_samples) < 20)
        length_of_padding = samples_per_manip - height(trimmed_pad_fil_stand_samples);
        padding = ones(length_of_padding, 6) .* trimmed_pad_fil_stand_samples(end, :);
        trimmed_pad_fil_stand_samples = vertcat(trimmed_pad_fil_stand_samples, padding);
    end

%     subplot(4, 1, 1);
%     plot(trimmed_pad);
%     subplot(4, 1, 2);
%     plot(trimmed_pad_fil);
%     subplot(4, 1, 3);
%     plot(trimmed_pad_fil_stand);
%     subplot(4, 1, 4);
%     plot(trimmed_pad_fil_stand_samples);

%     acc_calib = calib(:, 1:3);
%     gyro_calib = calib(:, 4:6);
    
    % Normalize to calibration
%     acc = acc(i0:i1, :) ./ acc_calib;
%     gyro = gyro(i0:i1, :) ./ gyro_calib;
%     acc_samples = acc(1:interval:end, :);
%     gyro_samples = gyro(1:interval:end, :);

%     figure();
%     subplot(2, 1, 1);
%     plot(acc(i0:i1, :));
%     subplot(2, 1, 2);
%     plot(gyro(i0:i1, :));

%     acc_samples = acc(i0:interval:i1, :);
%     acc_samples = acc_samples ./ max(abs(acc_samples));
%     gyro_samples = gyro(i0:interval:i1, :);
%     gyro_samples = gyro_samples ./ max(abs(gyro_samples));
    
    % Then Normalize to max?
%     acc_samples = acc_samples ./ max(abs(acc_samples), [], 1);
%     gyro_samples = gyro_samples ./ max(abs(gyro_samples), [], 1);

%     plot(acc_calib);
%     figure();
%     figure();
%     subplot(2, 1, 1);
%     plot(acc(i0:i1, :));
%     subplot(2, 1, 2);
%     plot(acc_samples);

%     subplot(
%     plot(acc);
%     figure();

%     trimmed_samples = reshape(trimmed_stand_fil_samples, [1, samples_per_manip*6*5]);

    trimmed_samples = reshape(trimmed_pad_fil_stand_samples, [1, samples_per_manip*6]);
    res = trimmed_samples;
  end


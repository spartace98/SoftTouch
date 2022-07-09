function [res] = get_window_intervals(data)
%GET_WINDOW_INTERVALS Summary of this function goes here
%   Get 200 points intervals
% result should be in the format [(start1, end1), (start2, end2)]

    interval = 30;
    res = zeros(4, 1);
    res(1) = data(1);
    j = 2;
    start = data(1);
    num_values = length(data);
    
%     drop values that are too close to each other
	for i=1:num_values
        if (data(i) - start > interval)
            res(j) = data(i);
            j = j + 1;
            start = data(i);
        else
            start = data(i);
        end
end


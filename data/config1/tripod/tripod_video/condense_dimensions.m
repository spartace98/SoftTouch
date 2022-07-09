function [condensed_data] = condense_dimensions(data)
%CONDENSE_DIMENSIONS Summary of this function goes here
%   Detailed explanation goes here

    num_sensors = 6;
    condensed_data = zeros(length(data), num_sensors);

    % new dimension every 6 columns
    for j=0:num_sensors-1
        cx = data(:, 1+2*j);
        cy = data(:, 2+2*j);
        dist = (cx.^2 + cy.^2).^0.5;
        condensed_data(:, j+1) = dist;
    end
end


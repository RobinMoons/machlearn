function [ featureMatrix ] = testFeatureExtraction( x_Measurement, y_Measurement, z_Measurement )
%TESTFEATUREEXTRACTION Summary of this function goes here
%   Detailed explanation goes here
featureMatrix = zeros(1,5);

% Resultant array
N = numel(x_Measurement);
result = zeros(N,1);
% Subtract mean from each value
for jj=1:N
    % Calculate resultant
    result(jj) = sqrt(x_Measurement(jj)^2 + y_Measurement(jj)^2 + z_Measurement(jj)^2);
end
mean_result = mean(result);
result = result - mean_result;
% Time domain waardes.
 featureMatrix(1,1) = mean_result;
%     featureMatrix(ii,1) = mean_result;
 featureMatrix(1,2) = std(result);
 featureMatrix(1,3) = skewness(result);
% Plot resultant in 1 subplot
% subplot(4,4,ii)
% plot(result)
% title(ii);

Fs = 128;      %sample frequentie
T = 1/Fs;       %sample periode
L = 1500;       %lengte van het signaal
Y = fft(result,L);
twoSideSpectrum = abs(Y/L);
singleSideSpectrum = twoSideSpectrum(1:L/2);
singleSideSpectrum(2:end-1) = 2*singleSideSpectrum(2:end-1);
f = Fs*(0:(L/2)-1)/L;
% Percentile berekening.    
cumsumgraph = cumsum(singleSideSpectrum.^2)/sum(singleSideSpectrum.^2);    
index25 = find(cumsumgraph >= 0.25, 1, 'first');
percentile25 = f(index25);    
index75 = find(cumsumgraph >= 0.75, 1, 'first');
percentile75 = f(index75); 
featureMatrix(1,4) = percentile25;
featureMatrix(1,5) = percentile75;
end


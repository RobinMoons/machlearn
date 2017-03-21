close all;
clear all;
clc;

load('data.mat')

%%Plot first elements of array
% --> niet meer nodig, verder berekenen we resultantes
% --> eventueel te gebruiken in verslag
% drinking_x_firstMeasurement = data.drinking(1).x;
% drinking_y_firstMeasurement = data.drinking(1).y;
% drinking_z_firstMeasurement = data.drinking(1).z;
% 
% figure, plot(drinking_x_firstMeasurement)
% 
% brush_x_firstMeasurement = data.brush(1).x;
% brush_y_firstMeasurement = data.brush(1).y;
% brush_z_firstMeasurement = data.brush(1).z;
% 
% figure, plot(brush_x_firstMeasurement)
% 
% shoe_x_firstMeasurement = data.shoe(1).x;
% shoe_y_firstMeasurement = data.shoe(1).y;
% shoe_z_firstMeasurement = data.shoe(1).z;
% 
% figure, plot(shoe_x_firstMeasurement)
% 
% writing_x_firstMeasurement = data.writing(1).x;
% writing_y_firstMeasurement = data.writing(1).y;
% writing_z_firstMeasurement = data.writing(1).z;
% 
% figure, plot(writing_x_firstMeasurement)

%% Removing the mean from the sample
%edit Robin: dit lijkt te werken maar hebben we hier veel aan?
%eventueel moeten we dit voor alle waardes doen en daarna de resultantes
%berekenen?

% edit Bram: Heb het nog snel wat aangepast, mijn for lus ging nog niet tot
% de lengte van de array. Had ik fout over gekopie�rd.
% En ik denk dat ook. Overal mean aftrekken en dan x,y,z samentellen tot 1
% resulterende vector?
% Ik heb eens geprobeerd om dit te berekenen voor alle 15 meetresultaten
% van drinking. Je kijkt maar eens wat er van klopt of hoe ik dingen beter
% kan doen in matlab :)

%edit Robin: goed gewerkt, enkel de formule voor de resultante is normaal
%toch: wortel(x�+y�+z�) ??
%de subplots geven mooi weer wat we nu net gaan zoeken,laat ze dus maar 
%staan voor in het verslag, die man vind dat goed als ge u redenering laat
%zien en we kunnen dat wel gebruiken. 



% Calculate drinking
figure('NumberTitle','off','Name','Drinking data')
amountDrinking = numel(data.drinking);
for ii=1:amountDrinking
    % Get the x,y,z measurements
    drinking_x_Measurement = data.drinking(ii).x;
    drinking_y_Measurement = data.drinking(ii).y;
    drinking_z_Measurement = data.drinking(ii).z;
    % Calculate mean
    drinking_x_mean = mean(drinking_x_Measurement);
    drinking_y_mean = mean(drinking_y_Measurement);
    drinking_z_mean = mean(drinking_z_Measurement);
    % Resultant array
    N = numel(drinking_x_Measurement);
    drinking_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
        drinking_x_Measurement(jj)=drinking_x_Measurement(jj)-drinking_x_mean;
        drinking_y_Measurement(jj)=drinking_y_Measurement(jj)-drinking_y_mean;
        drinking_z_Measurement(jj)=drinking_z_Measurement(jj)-drinking_z_mean;
        % Calculate resultant
        drinking_result(jj) = sqrt(drinking_x_Measurement(jj)^2 + drinking_y_Measurement(jj)^2 + drinking_z_Measurement(jj)^2);
    end
    mean_drinking_result = mean(drinking_result);
    drinking_result = drinking_result - mean_drinking_result;
    % Plot resultant in 1 subplot
    subplot(4,4,ii)
    plot(drinking_result)
    title(ii);
end


% Calculate brushing
figure('NumberTitle','off','Name','Brush data')
amountBrush = numel(data.brush);
for ii=1:amountBrush
    % Get the x,y,z measurements
    brush_x_Measurement = data.brush(ii).x;
    brush_y_Measurement = data.brush(ii).y;
    brush_z_Measurement = data.brush(ii).z;
    % Calculate mean
    brush_x_mean = mean(brush_x_Measurement);
    brush_y_mean = mean(brush_y_Measurement);
    brush_z_mean = mean(brush_z_Measurement);
    % Resultant array
    N = numel(brush_x_Measurement);
    brush_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
        brush_x_Measurement(jj)=brush_x_Measurement(jj)-brush_x_mean;
        brush_y_Measurement(jj)=brush_y_Measurement(jj)-brush_y_mean;
        brush_z_Measurement(jj)=brush_z_Measurement(jj)-brush_z_mean;
        % Calculate resultant
        brush_result(jj) = sqrt(brush_x_Measurement(jj)^2 + brush_y_Measurement(jj)^2 + brush_z_Measurement(jj)^2);
    end
    mean_brush_result = mean(brush_result);
    brush_result = brush_result - mean_brush_result;
    % Plot resultant in 1 subplot
    subplot(4,4,ii)
    plot(brush_result)
    title(ii);
end

% Calculate writing
figure('NumberTitle','off','Name','Writing data')
amountWriting = numel(data.writing);
for ii=1:amountWriting
    % Get the x,y,z measurements
    writing_x_Measurement = data.writing(ii).x;
    writing_y_Measurement = data.writing(ii).y;
    writing_z_Measurement = data.writing(ii).z;
    % Calculate mean
    writing_x_mean = mean(writing_x_Measurement);
    writing_y_mean = mean(writing_y_Measurement);
    writing_z_mean = mean(writing_z_Measurement);
    % Resultant array
    N = numel(writing_x_Measurement);
    writing_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
        writing_x_Measurement(jj)=writing_x_Measurement(jj)-writing_x_mean;
        writing_y_Measurement(jj)=writing_y_Measurement(jj)-writing_y_mean;
        writing_z_Measurement(jj)=writing_z_Measurement(jj)-writing_z_mean;
        % Calculate resultant
        writing_result(jj) = sqrt(writing_x_Measurement(jj)^2 + writing_y_Measurement(jj)^2 + writing_z_Measurement(jj)^2);
    end
    mean_writing_result = mean(writing_result);
    writing_result = writing_result - mean_writing_result;
    % Plot resultant in 1 subplot
    subplot(4,4,ii)
    plot(writing_result)
    title(ii);
end

% Calculate Shoe
amountShoe = numel(data.shoe);
figure('NumberTitle','off','Name','Shoe data')
for ii=1:amountShoe
    % Get the x,y,z measurements
    shoe_x_Measurement = data.shoe(ii).x;
    shoe_y_Measurement = data.shoe(ii).y;
    shoe_z_Measurement = data.shoe(ii).z;
    % Calculate mean
    shoe_x_mean = mean(shoe_x_Measurement);
    shoe_y_mean = mean(shoe_y_Measurement);
    shoe_z_mean = mean(shoe_z_Measurement);
    % Resultant array
    N = numel(shoe_x_Measurement);
    shoe_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
        shoe_x_Measurement(jj)=shoe_x_Measurement(jj)-shoe_x_mean;
        shoe_y_Measurement(jj)=shoe_y_Measurement(jj)-shoe_y_mean;
        shoe_z_Measurement(jj)=shoe_z_Measurement(jj)-shoe_z_mean;
        % Calculate resultant
        shoe_result(jj) = sqrt(shoe_x_Measurement(jj)^2 + shoe_y_Measurement(jj)^2 + shoe_z_Measurement(jj)^2);
    end
    mean_shoe_result = mean(shoe_result);
    shoe_result = shoe_result - mean_shoe_result;
    % Plot resultant in 1 subplot
    subplot(4,4,ii)
    plot(shoe_result)
    title(ii);
end

%% Feature extraction

%drinking
Fs = 128;      %sample frequentie
T = 1/Fs;       %sample periode
L = numel(drinking_result);       %lengte van het signaal
t = (0:L-1)*T;  %tijd vector

Y = fft(drinking_result,L);
twoSideSpectrum = abs(Y/L);
singleSideSpectrum = twoSideSpectrum(1:L/2+0);
singleSideSpectrum(2:end-1) = 2*singleSideSpectrum(2:end-1);
f = Fs*(0:(L/2)-1)/L;
figure('NumberTitle','off','Name','Drinking FFT result'), plot(f,singleSideSpectrum)
title('spectrum of the signal')
xlabel('f (Hz)')
ylabel('|singleSideSpectrum(f)|')

%brush
Fs = 128;      %sample frequentie
T = 1/Fs;       %sample periode
L = numel(brush_result);       %lengte van het signaal
t = (0:L-1)*T;  %tijd vector

Y = fft(brush_result,L);
twoSideSpectrum = abs(Y/L);
singleSideSpectrum = twoSideSpectrum(1:L/2+1);
singleSideSpectrum(2:end-1) = 2*singleSideSpectrum(2:end-1);
f = Fs*(0:(L/2))/L;
figure('NumberTitle','off','Name','Brush FFT result'), plot(f,singleSideSpectrum)
title('spectrum of the signal')
xlabel('f (Hz)')
ylabel('|singleSideSpectrum(f)|')


%writing
Fs = 128;      %sample frequentie
T = 1/Fs;       %sample periode
L = numel(writing_result);       %lengte van het signaal
t = (0:L-1)*T;  %tijd vector

Y = fft(writing_result,L);
twoSideSpectrum = abs(Y/L);
singleSideSpectrum = twoSideSpectrum(1:L/2+1);
singleSideSpectrum(2:end-1) = 2*singleSideSpectrum(2:end-1);
f = Fs*(0:(L/2))/L;
figure('NumberTitle','off','Name','Writing FFT result'), plot(f,singleSideSpectrum)
title('spectrum of the signal')
xlabel('f (Hz)')
ylabel('|singleSideSpectrum(f)|')

%Shoe
Fs = 128;      %sample frequentie
T = 1/Fs;       %sample periode
L = numel(shoe_result);       %lengte van het signaal
t = (0:L-1)*T;  %tijd vector

Y = fft(shoe_result,L);
twoSideSpectrum = abs(Y/L);
singleSideSpectrum = twoSideSpectrum(1:L/2+1);
singleSideSpectrum(2:end-1) = 2*singleSideSpectrum(2:end-1);
f = Fs*(0:(L/2))/L;
figure('NumberTitle','off','Name','Shoe FFT result'), plot(f,singleSideSpectrum)
title('spectrum of the signal')
xlabel('f (Hz)')
ylabel('|singleSideSpectrum(f)|')


%Time domain:
% -mean amplitude
% array maken van de amplitudes -> mean berekenen
% -standard deviation amplitude
% standaardafwijking berekenen op array van amplitudes
% -skewness: https://nl.mathworks.com/help/stats/skewness.html 

%Frequency domain:
% -


%% Scatter plots van de features
%info voor scattered plot
%https://nl.mathworks.com/help/matlab/ref/scatter.html 


%Nieuwe deel van de taak
%% Decision Trees for Binary Classification
% For illustration purpose use the 2 most discriminating features from the data exploration part. 
% Each group selects one of the four activities it wants to detect. 
% Use the binary classification approach one versus the rest to construct the decision tree. 
% Construct a decision tree with the training data given in data.mat. 
% You can use the instruction fitctree in MATLAB for this purpose. 
% Also divide the feature space in the region of the positive instances and the region of the negative instances. 
% Visualise also in the feature space the training instances. 
% Plot the Receiver-Operating-Characteristic (ROC) and also calculate the area-under-the-curve (AUC).
% Use for this purpose the MATLAB instructions perfcurve. 
% The score is obtained by the MATLAB instruction resubPredict for trees. 
% Also calculate the confusion matrix on the training set. Use for that the instruction confusionmat in MATLAB.  
% Extract the accuracy of the binary classifier. 
% Note that calculating performance measures on the training data gives too optimistic results (overfitting). 

















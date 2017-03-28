close all;
clear all;
clc;

testdata = load('testDataDetection.mat');
testdata = testdata.data;
data = load('data.mat');
data = data.data;



%%create featureMatrix
featureMatrix = zeros(numel(data.drinking) + numel(data.brush) + numel(data.shoe) + numel(data.writing),5);

%% Calculate drinking
figure('NumberTitle','off','Name','Drinking data')
amountDrinking = numel(data.drinking);

for ii=1:amountDrinking
    % Get the x,y,z measurements
    drinking_x_Measurement = data.drinking(ii).x;
    drinking_y_Measurement = data.drinking(ii).y;
    drinking_z_Measurement = data.drinking(ii).z;
    % Resultant array
    N = numel(drinking_x_Measurement);
    drinking_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
        % Calculate resultant
        drinking_result(jj) = sqrt(drinking_x_Measurement(jj)^2 + drinking_y_Measurement(jj)^2 + drinking_z_Measurement(jj)^2);
    end
    mean_drinking_result = mean(drinking_result);
    drinking_result = drinking_result - mean_drinking_result;
    % Time domain waardes.
    featureMatrix(ii,1) = mean_drinking_result;
%     featureMatrix(ii,1) = mean_drinking_result;
    featureMatrix(ii,2) = std(drinking_result);
    featureMatrix(ii,3) = skewness(drinking_result);
    % Plot resultant in 1 subplot
    subplot(4,4,ii)
    plot(drinking_result)
    title(ii);
    
    Fs = 128;      %sample frequentie
    T = 1/Fs;       %sample periode
    L = 1500;       %lengte van het signaal
    Y = fft(drinking_result,L);
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
    featureMatrix(ii,4) = percentile25;
    featureMatrix(ii,5) = percentile75;
end

%% Calculate brushing
figure('NumberTitle','off','Name','Brush data')
amountBrush = numel(data.brush);
for ii=1:amountBrush
    % Get the x,y,z measurements
    brush_x_Measurement = data.brush(ii).x;
    brush_y_Measurement = data.brush(ii).y;
    brush_z_Measurement = data.brush(ii).z;
    % Resultant array
    N = numel(brush_x_Measurement);
    brush_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
        % Calculate resultant
        brush_result(jj) = sqrt(brush_x_Measurement(jj)^2 + brush_y_Measurement(jj)^2 + brush_z_Measurement(jj)^2);
    end
    mean_brush_result = mean(brush_result);
    brush_result = brush_result - mean_brush_result;
    % Time domain waardes.
    featureMatrix(ii + amountDrinking,1) = mean_brush_result;
%     featureMatrix(ii + amountDrinking,1) = mean_brush_result;
    featureMatrix(ii + amountDrinking,2) = std(brush_result);
    featureMatrix(ii + amountDrinking,3) = skewness(brush_result);
    % Plot resultant in 1 subplot
    subplot(4,4,ii)
    plot(brush_result)
    title(ii);
    Fs = 128;      %sample frequentie
    T = 1/Fs;       %sample periode
    L = 1500;       %lengte van het signaal    
    Y = fft(brush_result,L);
    twoSideSpectrum = abs(Y/L);
    singleSideSpectrum = twoSideSpectrum(1:L/2);
    singleSideSpectrum(2:end-1) = 2*singleSideSpectrum(2:end-1);
    %percentile berekening
    cumsumgraph = cumsum(singleSideSpectrum.^2)/sum(singleSideSpectrum.^2);    
    index25 = find(cumsumgraph >= 0.25, 1, 'first');
    percentile25 = f(index25);    
    index75 = find(cumsumgraph >= 0.75, 1, 'first');
    percentile75 = f(index75); 
    featureMatrix(ii + amountDrinking, 4) = percentile25;
    featureMatrix(ii + amountDrinking, 5) = percentile75;
end

% Calculate writing
figure('NumberTitle','off','Name','Writing data')
amountWriting = numel(data.writing);
for ii=1:amountWriting
    % Get the x,y,z measurements
    writing_x_Measurement = data.writing(ii).x;
    writing_y_Measurement = data.writing(ii).y;
    writing_z_Measurement = data.writing(ii).z;
    % Resultant array
    N = numel(writing_x_Measurement);
    writing_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
        % Calculate resultant
        writing_result(jj) = sqrt(writing_x_Measurement(jj)^2 + writing_y_Measurement(jj)^2 + writing_z_Measurement(jj)^2);
    end
    mean_writing_result = mean(writing_result);
    writing_result = writing_result - mean_writing_result;
    featureMatrix(ii+ amountDrinking + amountBrush,1) = mean_writing_result;
    featureMatrix(ii+ amountDrinking + amountBrush,2) = std(writing_result);
    featureMatrix(ii+ amountDrinking + amountBrush,3) = skewness(writing_result);
    % Plot resultant in 1 subplot
    subplot(4,4,ii)
    plot(writing_result)
    title(ii);
    Fs = 128;      %sample frequentie
    T = 1/Fs;       %sample periode
    L = 1500;       %lengte van het signaal    
    Y = fft(writing_result,L);
    twoSideSpectrum = abs(Y/L);
    singleSideSpectrum = twoSideSpectrum(1:L/2);
    singleSideSpectrum(2:end-1) = 2*singleSideSpectrum(2:end-1);
    %berekening percentile
    cumsumgraph = cumsum(singleSideSpectrum.^2)/sum(singleSideSpectrum.^2);    
    index25 = find(cumsumgraph >= 0.25, 1, 'first');
    percentile25 = f(index25);    
    index75 = find(cumsumgraph >= 0.75, 1, 'first');
    percentile75 = f(index75); 
    featureMatrix(ii + amountDrinking + amountBrush,4) = percentile25;
    featureMatrix(ii + amountDrinking + amountBrush,5) = percentile75;
end

% Calculate Shoe
amountShoe = numel(data.shoe);
figure('NumberTitle','off','Name','Shoe data')
for ii=1:amountShoe
    % Get the x,y,z measurements
    shoe_x_Measurement = data.shoe(ii).x;
    shoe_y_Measurement = data.shoe(ii).y;
    shoe_z_Measurement = data.shoe(ii).z;
    % Resultant array
    N = numel(shoe_x_Measurement);
    shoe_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
        % Calculate resultant
        shoe_result(jj) = sqrt(shoe_x_Measurement(jj)^2 + shoe_y_Measurement(jj)^2 + shoe_z_Measurement(jj)^2);
    end
    mean_shoe_result = mean(shoe_result);
    shoe_result = shoe_result - mean_shoe_result;
    % Time features
    featureMatrix(ii+ amountDrinking + amountBrush + amountWriting,1) = mean_shoe_result;
    featureMatrix(ii+ amountDrinking + amountBrush + amountWriting,2) = std(shoe_result);
    featureMatrix(ii+ amountDrinking + amountBrush + amountWriting,3) = skewness(shoe_result);
    % Plot resultant in 1 subplot
    subplot(4,4,ii)
    plot(shoe_result)
    title(ii);
    
    Fs = 128;      %sample frequentie
    T = 1/Fs;       %sample periode
    L = 1500;       %lengte van het signaal

    Y = fft(shoe_result,L);
    twoSideSpectrum = abs(Y/L);
    singleSideSpectrum = twoSideSpectrum(1:L/2);
    singleSideSpectrum(2:end-1) = 2*singleSideSpectrum(2:end-1);
    f = Fs*(0:(L/2))/L;
    cumsumgraph = cumsum(singleSideSpectrum.^2)/sum(singleSideSpectrum.^2);    
    index25 = find(cumsumgraph >= 0.25, 1, 'first');
    percentile25 = f(index25);    
    index75 = find(cumsumgraph >= 0.75, 1, 'first');
    percentile75 = f(index75);    
    featureMatrix(ii + amountDrinking + amountBrush + amountWriting,4) = percentile25;
    featureMatrix(ii + amountDrinking + amountBrush + amountWriting,5) = percentile75;
end


%% Scatter plots van de features

Class = [ones(amountDrinking,1);2*ones(amountBrush + amountShoe + amountWriting,1)];
figure, gplotmatrix(featureMatrix,[],Class)

%Nieuwe deel van de taak
%% Decision Trees for Binary Classification
% For illustration purpose use the 2 most discriminating features from the data exploration part. 
% --> 25 en 75 percentile
% Each group selects one of the four activities it wants to detect. 
% --> drinking
% Use the binary classification approach one versus the rest to construct the decision tree. 
% Construct a decision tree with the training data given in data.mat. 
% You can use the instruction fitctree in MATLAB for this purpose. 

%% decission tree
tree = fitctree(featureMatrix, Class);
view(tree)
view(tree,'Mode','graph')
% Dit is niet nodig, de volledige featureMatrix kiest zelf al feature 4 en 5 uit.
% selectedFeatures = featureMatrix(: , 4:5);
% view(selectedTree)
% view(selectedTree,'Mode','graph')
% selectedTree = fitctree(selectedFeatures, Class);

%% Accuracy on trainings data
%help resubPredict
[Cpred_tr,score,node] = resubPredict(tree);
%help confusionmat
C = confusionmat(Class,Cpred_tr)
accuracy = trace(C)/sum(sum(C))



% Also divide the feature space in the region of the positive instances and the region of the negative instances. 
% Visualise also in the feature space the training instances. 
% Plot the Receiver-Operating-Characteristic (ROC) and also calculate the area-under-the-curve (AUC).
% Use for this purpose the MATLAB instructions perfcurve. 
% The score is obtained by the MATLAB instruction resubPredict for trees. 
% Also calculate the confusion matrix on the training set. Use for that the instruction confusionmat in MATLAB.  
% Extract the accuracy of the binary classifier. 
% Note that calculating performance measures on the training data gives too optimistic results (overfitting). 
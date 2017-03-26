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
    % Calculate mean
%     drinking_x_mean = mean(drinking_x_Measurement);
%     drinking_y_mean = mean(drinking_y_Measurement);
%     drinking_z_mean = mean(drinking_z_Measurement);
    % Resultant array
    N = numel(drinking_x_Measurement);
    drinking_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
%         drinking_x_Measurement(jj)=drinking_x_Measurement(jj)-drinking_x_mean;
%         drinking_y_Measurement(jj)=drinking_y_Measurement(jj)-drinking_y_mean;
%         drinking_z_Measurement(jj)=drinking_z_Measurement(jj)-drinking_z_mean;
        % Calculate resultant
        drinking_result(jj) = sqrt(drinking_x_Measurement(jj)^2 + drinking_y_Measurement(jj)^2 + drinking_z_Measurement(jj)^2);
    end
    mean_drinking_result = mean(drinking_result);
    drinking_result = drinking_result - mean_drinking_result;
    % Time domain waardes.
    featureMatrix(ii,1) = mean(drinking_result);
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
    % figure('NumberTitle','off','Name','Drinking FFT result'), plot(f,singleSideSpectrum)
    % title('spectrum of the signal')
    % xlabel('f (Hz)')
    % ylabel('|singleSideSpectrum(f)|')
    % Percentile berekening.
    percentile25 = prctile(singleSideSpectrum,25);
    percentile75 = prctile(singleSideSpectrum,75);
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
    % Calculate mean
    %brush_x_mean = mean(brush_x_Measurement);
    %brush_y_mean = mean(brush_y_Measurement);
    %brush_z_mean = mean(brush_z_Measurement);
    % Resultant array
    N = numel(brush_x_Measurement);
    brush_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
        %brush_x_Measurement(jj)=brush_x_Measurement(jj)-brush_x_mean;
        %brush_y_Measurement(jj)=brush_y_Measurement(jj)-brush_y_mean;
        %brush_z_Measurement(jj)=brush_z_Measurement(jj)-brush_z_mean;
        % Calculate resultant
        brush_result(jj) = sqrt(brush_x_Measurement(jj)^2 + brush_y_Measurement(jj)^2 + brush_z_Measurement(jj)^2);
    end
    mean_brush_result = mean(brush_result);
    brush_result = brush_result - mean_brush_result;
    % Time domain waardes.
    featureMatrix(ii + amountDrinking,1) = mean(brush_result);
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
%     f = Fs*(0:(L/2))/L;
%     figure('NumberTitle','off','Name','Writing FFT result'), plot(f,singleSideSpectrum)
%     title('spectrum of the signal')
%     xlabel('f (Hz)')
%     ylabel('|singleSideSpectrum(f)|')
    percentile25 = prctile(singleSideSpectrum,25);
    percentile75 = prctile(singleSideSpectrum,75);
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
    % Calculate mean
%     writing_x_mean = mean(writing_x_Measurement);
%     writing_y_mean = mean(writing_y_Measurement);
%     writing_z_mean = mean(writing_z_Measurement);
    % Resultant array
    N = numel(writing_x_Measurement);
    writing_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
%         writing_x_Measurement(jj)=writing_x_Measurement(jj)-writing_x_mean;
%         writing_y_Measurement(jj)=writing_y_Measurement(jj)-writing_y_mean;
%         writing_z_Measurement(jj)=writing_z_Measurement(jj)-writing_z_mean;
        % Calculate resultant
        writing_result(jj) = sqrt(writing_x_Measurement(jj)^2 + writing_y_Measurement(jj)^2 + writing_z_Measurement(jj)^2);
    end
    mean_writing_result = mean(writing_result);
    writing_result = writing_result - mean_writing_result;
    featureMatrix(ii+ amountDrinking + amountBrush,1) = mean(writing_result);
%     featureMatrix(ii+ amountDrinking + amountBrush,1) = mean_writing_result;
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
%     f = Fs*(0:(L/2))/L;
%     figure('NumberTitle','off','Name','Writing FFT result'), plot(f,singleSideSpectrum)
%     title('spectrum of the signal')
%     xlabel('f (Hz)')
%     ylabel('|singleSideSpectrum(f)|')
    percentile25 = prctile(singleSideSpectrum,25);
    percentile75 = prctile(singleSideSpectrum,75);
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
    % Calculate mean
%     shoe_x_mean = mean(shoe_x_Measurement);
%     shoe_y_mean = mean(shoe_y_Measurement);
%     shoe_z_mean = mean(shoe_z_Measurement);
    % Resultant array
    N = numel(shoe_x_Measurement);
    shoe_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
%         shoe_x_Measurement(jj)=shoe_x_Measurement(jj)-shoe_x_mean;
%         shoe_y_Measurement(jj)=shoe_y_Measurement(jj)-shoe_y_mean;
%         shoe_z_Measurement(jj)=shoe_z_Measurement(jj)-shoe_z_mean;
        % Calculate resultant
        shoe_result(jj) = sqrt(shoe_x_Measurement(jj)^2 + shoe_y_Measurement(jj)^2 + shoe_z_Measurement(jj)^2);
    end
    mean_shoe_result = mean(shoe_result);
    shoe_result = shoe_result - mean_shoe_result;
    % Time features
    featureMatrix(ii+ amountDrinking + amountBrush + amountWriting,1) = mean(shoe_result);
%     featureMatrix(ii+ amountDrinking + amountBrush + amountWriting,1) = mean_shoe_result;
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
%     figure('NumberTitle','off','Name','Shoe FFT result'), plot(f,singleSideSpectrum)
%     title('spectrum of the signal')
%     xlabel('f (Hz)')
%     ylabel('|singleSideSpectrum(f)|')
    percentile25 = prctile(singleSideSpectrum,25);
    percentile75 = prctile(singleSideSpectrum,75);
    featureMatrix(ii + amountDrinking + amountBrush + amountWriting,4) = percentile25;
    featureMatrix(ii + amountDrinking + amountBrush + amountWriting,5) = percentile75;
end





%Time domain:
% -mean amplitude
% array maken van de amplitudes -> mean berekenen
% -standard deviation amplitude
% standaardafwijking berekenen op array van amplitudes
% -skewness: https://nl.mathworks.com/help/stats/skewness.html 

%Frequency domain:
% -


%% Scatter plots van de features

Class = [ones(amountDrinking,1);2*ones(amountBrush + amountShoe + amountWriting,1)];
figure, gplotmatrix(featureMatrix,[],Class)

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

%% Overbodig 
% % Aantal elementen in de array / aantal features
% n = numel(featureMatrix) / 5;
% % Code vanuit voorbeeld -> gaat out of bounds
% %p = randperm(2*n)
% % deze werkt wel
% p = randperm(n);
% % Hier ook 2*n verwijderd
% Xte = featureMatrix(p(n+1:n),:);
% Clte = Class(p(n+1:n));
% %Training set 50% of data
% Xtr = featureMatrix(p(1:n),:);
% Cltr = Class(p(1:n));

%help fitctree
%Mdl = fitctree(Xtr(:,1:2),Cltr);

%% tree maken
tree = fitctree(featureMatrix, Class);
view(tree)
view(tree,'Mode','graph')

%%
% die voorbeeldcode daar heeft die eerst nog data moeten genereren enzo

% Code van het voorbeeld -> Geeft conversion error omdat de waardes allemaal doubles zijn 
% View feature space split in two classes
% %help meshgrid
% d = 0.01;
% [x1Grid,x2Grid] = meshgrid(min(Xtr(:,1)):d:max(Xtr(:,1)),...
%     min(Xtr(:,2)):d:max(Xtr(:,2)));
% xGrid = [x1Grid(:),x2Grid(:)];
% labels = predict(Mdl,xGrid);
% 
% % Training data points
% figure
% h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
% hold on
% h(3:4) = gscatter(Xtr(:,1),Xtr(:,2),Cltr);
% legend(h,{'Class1','Class2','Class1 Tr','Class2 Tr'},...
%    'Location','Northwest');
% xlabel('x1');
% ylabel('x2');
% 
% % Testing data points
% figure 
% 
% h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
% hold on
% h(3:4) = gscatter(Xte(:,1),Xte(:,2),Clte);
% legend(h,{'Class1','Class2','Class1 Te','Class2 Te'},...
%    'Location','Northwest');
% xlabel('x1');
% ylabel('x2');
% 
% % ROC curve one vs one
% % help resubPredict
% % [~,score] = resubPredict(Mdl);
% % Class1 vs Class2
% %help perfcurve
% [fpr,tpr,T,AUC,OPTROCPT] = perfcurve(Cltr,score(:,1),1);
% figure
% plot(fpr,tpr)
% hold on
% plot(OPTROCPT(1),OPTROCPT(2),'ro')
% xlabel('False positive rate')
% ylabel('True positive rate')
% title('ROC Curve for Classification by Classification Trees')
% hold off
% 
% % Data preparation for classification app
% 
% Y = [featureMatrix Class];















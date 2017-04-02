close all;
clear all;
clc;

testdata = load('testDataDetection.mat');
testdata = testdata.data;
data = load('data.mat');
data = data.data;

%% Extract features 
drinkingFeature = featureExtraction(data.drinking);
brushingFeature = featureExtraction(data.brush);
writingFeature = featureExtraction(data.writing);
shoeFeature = featureExtraction(data.shoe);
featureMatrix = [drinkingFeature; brushingFeature; writingFeature; shoeFeature];

%% Scatter plots van de features
amountDrinking = numel(data.drinking);
amountBrush = numel(data.brush);
amountShoe = numel(data.shoe);
amountWriting = numel(data.writing);
Class = [ones(amountDrinking,1);2*ones(amountBrush + amountShoe + amountWriting,1)];
figure, gplotmatrix(featureMatrix,[],Class)
title('gplotmatrix featureMatrix')

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
accuracyTraingData = trace(C)/sum(sum(C))


%% Inlezen testdata
testDataX = testdata.AthensTest_Accel_LN_X_CAL;
testDataY = testdata.AthensTest_Accel_LN_Y_CAL;
testDataZ = testdata.AthensTest_Accel_LN_Z_CAL;
testDataTime = testdata.AthensTest_Timestamp_Unix_CAL;
testDataLabel = testdata.Label;
figure, plot(testDataTime,testDataLabel)
title('test Data')

testDataSize = numel(testDataTime);
% Verdeling moet nog beter gebeuren, nu wordt de laatste kolom gevuld met
% nullen.
seg_length = round(testDataSize/80, -2);
% Elke Seg matrix heeft per kolom 1 data segmentatie.
timeSeg = zeros(seg_length,ceil(testDataSize/seg_length));
timeSeg(1:testDataSize) = testDataTime(:);
labelSeg = zeros(seg_length,ceil(testDataSize/seg_length));
labelSeg(1:testDataSize) = testDataTime(:);
xSeg = zeros(seg_length,ceil(testDataSize/seg_length));
xSeg(1:testDataSize) = testDataTime(:);
ySeg = zeros(seg_length,ceil(testDataSize/seg_length));
ySeg(1:testDataSize) = testDataTime(:);
zSeg = zeros(seg_length,ceil(testDataSize/seg_length));
zSeg(1:testDataSize) = testDataTime(:);


% Verwerken data
koloms = numel(timeSeg)/seg_length;
testFeatureMatrix = zeros(koloms, 5);
for ii=1:koloms
    % Feature extraction: 
    tempFeatureMatrix = testFeatureExtraction(xSeg(:,ii), ySeg(:,ii), zSeg(:,ii));
    testFeatureMatrix(ii,1) = tempFeatureMatrix(1,1);
    testFeatureMatrix(ii,2) = tempFeatureMatrix(1,2);
    testFeatureMatrix(ii,3) = tempFeatureMatrix(1,3);
    testFeatureMatrix(ii,4) = tempFeatureMatrix(1,4);
    testFeatureMatrix(ii,5) = tempFeatureMatrix(1,5);
end

%% Test desicion tree
Cpred = predict(tree,testFeatureMatrix);
% van vb Clte = Class(p(n+1:2*n));
% Deze getallen gebruikt omdat de som 81 is wat gelijk is aan het aantal
% rijen van de testFeatureMatrix omdat de class van trainingsdata ook
% gelijk is aantal rijen featurematrix.
ClassTest = [ones(21,1);2*ones(20 + 20 + 20,1)];
Clte = ClassTest;
% Accurcy
C = confusionmat(Clte,Cpred)
accuracyTestData = trace(C)/sum(sum(C))


% Also divide the feature space in the region of the positive instances and the region of the negative instances. 
% Visualise also in the feature space the training instances. 
% Plot the Receiver-Operating-Characteristic (ROC) and also calculate the area-under-the-curve (AUC).
% Use for this purpose the MATLAB instructions perfcurve. 
% The score is obtained by the MATLAB instruction resubPredict for trees. 
% Also calculate the confusion matrix on the training set. Use for that the instruction confusionmat in MATLAB.  
% Extract the accuracy of the binary classifier. 
% Note that calculating performance measures on the training data gives too optimistic results (overfitting). 
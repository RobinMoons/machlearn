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
col1 = [drinkingFeature(:,4); brushingFeature(:,4); writingFeature(:,4); shoeFeature(:,4)];
col2 = [drinkingFeature(:,5); brushingFeature(:,5); writingFeature(:,5); shoeFeature(:,5)];
featureMatrix = [col1,col2];
%TestfeatureMatrix = featureMatrix';
   
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

%% Visualisation of results
d = 0.01;
[x1Grid,x2Grid] = meshgrid(min(featureMatrix(:,1)):d:max(featureMatrix(:,1)),min(featureMatrix(:,2)):d:max(featureMatrix(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
labels = predict(tree,xGrid);
figure, h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
hold on
h(3:4) = gscatter(featureMatrix(:,1),featureMatrix(:,2),Class);
legend(h,{'Class1','Class2','Class1 Tr','Class2 Tr'},'Location','Northwest');
xlabel('x1');
ylabel('x2');

%% Accuracy on trainings data
%help resubPredict
[Cpred_tr,score,node] = resubPredict(tree);
%help confusionmat
C = confusionmat(Class,Cpred_tr)
accuracyTraingData = trace(C)/sum(sum(C))

%%
%%dit is verkeerd, we moeten niet met tijdsegmenatie werken maar de
%%segmentatie moet gebeuren met de labels

% testDataSize = numel(testDataTime);
% % Verdeling moet nog beter gebeuren, nu wordt de laatste kolom gevuld met
% % nullen.
% seg_length = round(testDataSize/80, -2);
% % Elke Seg matrix heeft per kolom 1 data segmentatie.
% timeSeg = zeros(seg_length,ceil(testDataSize/seg_length));
% timeSeg(1:testDataSize) = testDataTime(:);
% labelSeg = zeros(seg_length,ceil(testDataSize/seg_length));
% labelSeg(1:testDataSize) = testDataLabel(:);
% xSeg = zeros(seg_length,ceil(testDataSize/seg_length));
% xSeg(1:testDataSize) = testDataX(:);
% ySeg = zeros(seg_length,ceil(testDataSize/seg_length));
% ySeg(1:testDataSize) = testDataY(:);
% zSeg = zeros(seg_length,ceil(testDataSize/seg_length));
% zSeg(1:testDataSize) = testDataZ(:);


%%
%data segmentatie

vorige = -1;
tellerMeetpunt = 0;
activiteitenTeller = 0;
testDataLabel = testdata.Label;
for i = 1:length(testDataLabel)
    %% Inlezen testdata
    testDataX = testdata.AthensTest_Accel_LN_X_CAL(i);
    testDataY = testdata.AthensTest_Accel_LN_Y_CAL(i);
    testDataZ = testdata.AthensTest_Accel_LN_Z_CAL(i);
    testDataLabel = testdata.Label(i);
   if (testDataLabel ~= vorige)
       if (activiteitenTeller ~= 0)
           testActiviteiten(activiteitenTeller).x(tellerMeetpunt) = testDataX.'; 
           testActiviteiten(activiteitenTeller).y(tellerMeetpunt) = testDataY.'; 
           testActiviteiten(activiteitenTeller).z(tellerMeetpunt) = testDataZ.'; 
           testActiviteiten(activiteitenTeller).label(tellerMeetpunt) = testDataLabel.';
       end
       activiteitenTeller = activiteitenTeller + 1;
       tellerMeetpunt = 0;
   end
   vorige = testDataLabel;
   tellerMeetpunt = tellerMeetpunt + 1;
   
   testActiviteiten(activiteitenTeller).x(tellerMeetpunt) = testDataX; 
   testActiviteiten(activiteitenTeller).y(tellerMeetpunt) = testDataY; 
   testActiviteiten(activiteitenTeller).z(tellerMeetpunt) = testDataZ; 
   testActiviteiten(activiteitenTeller).label(tellerMeetpunt) = testDataLabel;
end

%%
% Verwerken data
testFeatureMatrix = featureExtraction(testActiviteiten);
%% Test desicion tree

%% Accuracy on trainings data

Cpred = predict(tree,[testFeatureMatrix(:,4), testFeatureMatrix(:,5)]);
% van vb Clte = Class(p(n+1:2*n));
% Deze getallen gebruikt omdat de som 81 is wat gelijk is aan het aantal
% rijen van de testFeatureMatrix omdat de class van trainingsdata ook
% gelijk is aantal rijen featurematrix.

%%ROBIN: volgens mij moeten we nu een class maken door alle activiteiten
%%met label '1' (als dat drinking is) "1" te maken, en al de resst "2" maar
%%hier had ik geen tijd meer voor :)
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
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

%% Visualisation of results
% d = 0.01;
% [x1Grid,x2Grid] = meshgrid(min(featureMatrix(:,1)):d:max(featureMatrix(:,1)),min(featureMatrix(:,2)):d:max(featureMatrix(:,2)));
% xGrid = [x1Grid(:),x2Grid(:)];
% labels = predict(tree,xGrid);
% figure, h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
% hold on
% h(3:4) = gscatter(featureMatrix(:,1),featureMatrix(:,2),Class);
% legend(h,{'Class1','Class2','Class1 Tr','Class2 Tr'},'Location','Northwest');
% xlabel('x1');
% ylabel('x2');

%% Accuracy on trainings data
%help resubPredict
[Cpred_tr,score,node] = resubPredict(tree);
%help confusionmat
C = confusionmat(Class,Cpred_tr)
accuracyTraingData = trace(C)/sum(sum(C))

%%
%data segmentatie
vorige = -1;
tellerMeetpunt = 0;
activiteitenTeller = 0;
testDataLabel = testdata.Label;
% drinkingTeller = 0;
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
%        if(testDataLabel == 1)
%            drinkingTeller = drinkingTeller + 1;
%        end
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
Cpred = predict(tree,[testFeatureMatrix(:,4), testFeatureMatrix(:,5)]);


%% Accuracy on trainings data
total = numel(testFeatureMatrix(:,4));

%ClassTest = [ones(drinkingTeller,1);2*ones(total-drinkingTeller,1)];
ClassTest = [];
for i = 1 : total
    if (testActiviteiten(i).label(1) == 1)
        ClassTest = vertcat(ClassTest, 1);
    else
        ClassTest = vertcat(ClassTest, 2);
    end
end

Clte = ClassTest;

%% Visualisation of results
% help meshgrid
d = 0.01;
[x1Grid,x2Grid] = meshgrid(min(featureMatrix(:,1)):d:max(featureMatrix(:,1)),...
    min(featureMatrix(:,2)):d:max(featureMatrix(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

labels = predict(tree,xGrid);

% Training data points
figure('Name', 'Division 2D feature space trainingsdata', 'NumberTitle', 'off')
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
hold on
h(3:4) = gscatter(featureMatrix(:,1),featureMatrix(:,2),Class);
legend(h,{'Class1','Class2','Class1 Tr','Class2 Tr'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');

% Testing data points
figure('Name', 'Division 2D feature space testdata (from testData.mat)', 'NumberTitle', 'off')
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
hold on
h(3:4) = gscatter(testFeatureMatrix(:,4),testFeatureMatrix(:,5),Clte);
legend(h,{'Class1','Class2','Class1 Te','Class2 Te'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');

%% Accurcy
C = confusionmat(Clte,Cpred)
accuracyTestData = trace(C)/sum(sum(C))

%% Bram: verder gegaan met de voorbeeldcode van de prof.
% ROC curve one vs one
% help resubPredict
[~,score] = resubPredict(tree);
%Class1 vs Class2
%help perfcurve
[fpr,tpr,T,AUC,OPTROCPT] = perfcurve(Class,score(:,1),1);
AUC
figure('Name', 'ROC curve testdata (from testData.mat)', 'NumberTitle', 'off')
plot(fpr,tpr)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by Classification Trees')
hold off

% Also divide the feature space in the region of the positive instances and the region of the negative instances. 
% Visualise also in the feature space the training instances. 
% Plot the Receiver-Operating-Characteristic (ROC) and also calculate the area-under-the-curve (AUC).
% Use for this purpose the MATLAB instructions perfcurve. 
% The score is obtained by the MATLAB instruction resubPredict for trees. 
% Also calculate the confusion matrix on the training set. Use for that the instruction confusionmat in MATLAB.  
% Extract the accuracy of the binary classifier. 
% Note that calculating performance measures on the training data gives too optimistic results (overfitting). 
close all;
clear all;
clc;

%Selected activity: DRINKING
%Explanation of variables
% featureMatrix_s : features from the whole small dataset
% featureMatrix_l : features from the whole large dataset
% Class_s : class from the whole small dataset (chosen DRINKING)
% Class_l : class from the whole large dataset (chosen DRINKING)

%% load datasets
%load large dataset
largeData = load('testDataDetection.mat');
largeData = largeData.data;
%load small dataset
smallData = load('data.mat');
smallData = smallData.data;

%% Extract features from small data set 
drinkingFeature = featureExtraction(smallData.drinking);
brushingFeature = featureExtraction(smallData.brush);
writingFeature = featureExtraction(smallData.writing);
shoeFeature = featureExtraction(smallData.shoe);
col1 = [drinkingFeature(:,4); brushingFeature(:,4); writingFeature(:,4); shoeFeature(:,4)];
col2 = [drinkingFeature(:,5); brushingFeature(:,5); writingFeature(:,5); shoeFeature(:,5)];
featureMatrix_s = [col1,col2];
%create class (used to check the results)
amountDrinking = numel(smallData.drinking);
amountBrush = numel(smallData.brush);
amountShoe = numel(smallData.shoe);
amountWriting = numel(smallData.writing);
Class_s = [ones(amountDrinking,1);2*ones(amountBrush + amountShoe + amountWriting,1)];   
% Scatter plots from features 
figure, gplotmatrix(featureMatrix_s,[],Class_s)
title('gplotmatrix featureMatrix_s')

%% Extract features from 1/3 and 2/3 dataset from small dataset
%create array with 4 activities from each feature ( = approx. 1/3 of the set)
activities_1_3 = [smallData.drinking(1:4),smallData.writing(1:4),smallData.shoe(1:4),smallData.brush(1:4)]; 
%create array with the rest of the activities from each feature ( = approx. 2/3 of the set)
activities_2_3 = [smallData.drinking(5:amountDrinking),smallData.writing(5:amountWriting),smallData.shoe(5:amountShoe),smallData.brush(5:amountBrush)];
number_1_3 = numel(activities_1_3);
number_2_3 = numel(activities_2_3);
%Create class for 1/3 set 1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,
Class_1_3 = [ones(4,1);2*ones(number_1_3-4,1)]; 
%Create class with the rest of drinking as 1 and the other rest as 2
Class_2_3 = [ones(amountDrinking-4,1);2*ones(number_2_3-(amountDrinking-4),1)]; 
%extract features from 1/3 set
featureMatrix_1_3 = featureExtraction(activities_1_3);
%extract features from 2/3 set
featureMatrix_2_3 = featureExtraction(activities_2_3);

%% Extract features from large data set
%segmentatie large dataset
numberSamples = numel(largeData.AthensTest_Accel_LN_X_CAL)
size = 2000;
numberActivities = floor(numberSamples / size)
drinkingActivityCounter = 0;
for activity = 1:1:numberActivities
    drinkingCounter = 0;
    for i = 1:1:size
        testDataX = largeData.AthensTest_Accel_LN_X_CAL((activity-1)*size + i);
        testDataY = largeData.AthensTest_Accel_LN_Y_CAL((activity-1)*size + i);
        testDataZ = largeData.AthensTest_Accel_LN_Z_CAL((activity-1)*size + i);        
        testDataLabel = largeData.Label((activity-1)*size + i);
        testActiviteiten(activity).x(i) = testDataX.';
        testActiviteiten(activity).y(i) = testDataY.';
        testActiviteiten(activity).z(i) = testDataZ.';
        testActiviteiten(activity).label(i) = testDataLabel.';
        if (testDataLabel == 1) %activity drinking = 1
            drinkingCounter = drinkingCounter + 1;        
        end 
    end
    if (drinkingCounter > (size/2))
        drinkingActivityCounter = drinkingActivityCounter +1;
    end
end
%extract features
featureMatrix_l = featureExtraction(testActiviteiten);
featureMatrix_l = [featureMatrix_l(:,4),featureMatrix_l(:,5)];
%create class (used to check the results)
Class_l = [ones(drinkingActivityCounter,1);2*ones((numberActivities - drinkingActivityCounter),1)];    
%Scatter plots from features 
%figure, gplotmatrix(featureMatrix_l,[],Class_l);
%title('gplotmatrix featureMatrix_l')

%% Binary classification
%train with large dataset
tree = fitctree(featureMatrix_s, Class_s);
%view(tree)
%view(tree,'Mode','graph')
Cpred = predict(tree,[featureMatrix_l(:,4), featureMatrix_l(:,5)]);
% Check accuracy 
[Cpred_tr,score,node] = resubPredict(tree);
C_decision_tr = confusionmat(Class_s,Cpred_tr)
accuracySmallData = trace(C_decision_tr)/sum(sum(C_decision_tr))

%train with small dataset


%% Visualisation of results
% help meshgrid
d = 0.01;
[x1Grid,x2Grid] = meshgrid(min(featureMatrix_s(:,1)):d:max(featureMatrix_s(:,1)),...
    min(featureMatrix_s(:,2)):d:max(featureMatrix_s(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
labels = predict(tree,xGrid);
% Training data points
figure('Name', 'Division 2D feature space trainingsdata', 'NumberTitle', 'off')
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
hold on
h(3:4) = gscatter(featureMatrix_s(:,1),featureMatrix_s(:,2),Class_s);
legend(h,{'Class1','Class2','Class1 Tr','Class2 Tr'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');
% Testing data points
figure('Name', 'Division 2D feature space testdata (from testData.mat)', 'NumberTitle', 'off')
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
hold on
h(3:4) = gscatter(featureMatrix_l(:,4),featureMatrix_l(:,5),Clte);
legend(h,{'Class1','Class2','Class1 Te','Class2 Te'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');

%% Accurcy
C_decision_te = confusionmat(Clte,Cpred)
accuracyTestData = trace(C_decision_te)/sum(sum(C_decision_te))

%% Bram: verder gegaan met de voorbeeldcode van de prof.
% ROC curve one vs one
% help resubPredict
[~,score] = resubPredict(tree);
%Class1 vs Class2
%help perfcurve
[fpr,tpr,T,AUC,OPTROCPT] = perfcurve(Class_s,score(:,1),1);
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

%% SVM one vs the rest code.
% Overgenomen van voorbeeld code die wel one vs one is.

% SVM classifier one vs one

%help fitcsvm
%SVMModel = fitcsvm(Xtr(:,1:2),Cltr,'OptimizeHyperparameters','auto');
SVMModel = fitcsvm(featureMatrix_s(:,1:2),Class_s);
% Accuracy on trainings data
%help resubPredict
[Cpred_tr,score,node] = resubPredict(SVMModel);
%help confusionmat
C_SVM_tr = confusionmat(Class_s,Cpred_tr)
accuracy_SVM_tr = trace(C_SVM_tr)/sum(sum(C_SVM_tr))

% Accuracy on test data
%help confusionmat
[Cpred,score] = predict(SVMModel,featureMatrix_l(:,4:5));
C_SVM_te = confusionmat(Clte,Cpred)
accuracy_SVM_te = trace(C_SVM_te)/sum(sum(C_SVM_te))

% visualisation of results

%help meshgrid
d = 0.01;
[x1Grid,x2Grid] = meshgrid(min(featureMatrix_s(:,1)):d:max(featureMatrix_s(:,1)),...
    min(featureMatrix_s(:,2)):d:max(featureMatrix_s(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

labels = predict(SVMModel,xGrid);

% Training data points
figure('Name', 'SVM - 2D division feature curve trainingsdata', 'NumberTitle', 'off')
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
hold on
h(3:4) = gscatter(featureMatrix_s(:,1),featureMatrix_s(:,2),Class_s);
legend(h,{'Class1','Class2','Class1 Tr','Class2 Tr'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');

% Testing data points
figure('Name', 'SVM - 2D division feature testdata (from testData.mat)', 'NumberTitle', 'off')
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
hold on
h(3:4) = gscatter(featureMatrix_l(:,4),featureMatrix_l(:,5),Clte);
legend(h,{'Class1','Class2','Class1 Te','Class2 Te'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');

% ROC curve one vs one
[fpr,tpr,T,AUC,OPTROCPT] = perfcurve(Clte,score(:,1),1);
AUC
figROC=figure('Name', 'SVM - ROC curve testdata (from testData.mat)', 'NumberTitle', 'off')
plot(fpr,tpr,'.-')
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by Classification SVM linear')

%% Naive bayes as Binary classificcation
% Meshgrid
d = 0.1;
[x1Grid,x2Grid] = meshgrid(min(featureMatrix_s(:,1)):d:max(featureMatrix_s(:,1)),...
    min(featureMatrix_s(:,2)):d:max(featureMatrix_s(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

% Classification
bayesTree = fitcnb(featureMatrix_s,Class_s);

bayesTree
bayesTree.DistributionParameters
Params = cell2mat(bayesTree.DistributionParameters);

Mu = Params([1 3],1:2); % Extract the means
Sigma = zeros(2,2,3);

face = {'r' , 'b' }

for j = 1:2
    Sigma(:,:,j) = diag(Params(2*j,:)).^2; % Create diagonal covariance matrix
    
    surf(x1Grid,x2Grid,reshape(mvnpdf(xGrid,Mu(j,:),Sigma(:,:,j)),size(x1Grid)),'FaceAlpha',0.5,'FaceColor',face{j})
        % Draw contours for the multivariate normal distributions
end

% Accuracy on trainings data
%help resubPredict
[Cpred_tr,score,node] = resubPredict(bayesTree);
%help confusionmat
C_bayes_tr = confusionmat(Class_s,Cpred_tr)
accuracy = trace(C_bayes_tr)/sum(sum(C_bayes_tr))

[Cpred,score] = predict(bayesTree,featureMatrix_l(:,4:5));
C_bayes_te = confusionmat(Clte,Cpred)
accuracy = trace(C_bayes_te)/sum(sum(C_bayes_te))

% visualisation of results
labels = predict(bayesTree,xGrid);

% Training data points
figure('Name', 'Naive Bayes - 2D division feature curve trainingsdata', 'NumberTitle', 'off')
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
hold on
h(3:4) = gscatter(featureMatrix_s(:,1),featureMatrix_s(:,2),Class_s);
legend(h,{'Class1','Class2','Class1 Tr','Class2 Tr'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');

% Testing data points
figure('Name', 'Naive Bayes - 2D division feature curve testdata (from testData.mat)', 'NumberTitle', 'off')
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
hold on
h(3:4) = gscatter(featureMatrix_l(:,4),featureMatrix_l(:,5),Clte);
legend(h,{'Class1','Class2','Class1 Te','Class2 Te'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');

% ROC curve one vs one
[fpr,tpr,T,AUC,OPTROCPT] = perfcurve(Clte,score(:,1),1);
AUC
figROC=figure;
plot(fpr,tpr,'.-')
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by Naive Bayes')

%% K-nearest neighbour as Binary Classification
% meshgrid
d = 0.1;
[x1Grid,x2Grid] = meshgrid(min(featureMatrix_s(:,1)):d:max(featureMatrix_s(:,1)),...
    min(featureMatrix_s(:,2)):d:max(featureMatrix_s(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

% knn Classifier k == 1 dist = euclidian
% Voor: Mdl = fitcknn(Xtr,Cltr);

% knn Classifier k == opt dist = opt

%werkend bij Bram
knnTree = fitcknn(featureMatrix_s,Class_s,'OptimizeHyperparameters','auto')

%knnTree = fitcknn(featureMatrix,Class,'Standardize','on')

knnTree

% Accuracy on trainings data
[Cpred_tr,score,node] = resubPredict(knnTree);
C_knear_tr = confusionmat(Class_s,Cpred_tr)
accuracy = trace(C_knear_tr)/sum(sum(C_knear_tr))

% Accuracy on test data
[Cpred,score] = predict(knnTree,featureMatrix_l(:,4:5));
C_knear_te = confusionmat(Clte,Cpred)
accuracy = trace(C_knear_te)/sum(sum(C_knear_te))

% visualisation of results
labels = predict(knnTree,xGrid);

% Training data points
figure('Name', 'K-nearest neighbor - 2D division feature curve trainingsdata', 'NumberTitle', 'off')
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
hold on
h(3:4) = gscatter(featureMatrix_s(:,1),featureMatrix_s(:,2),Class_s);
legend(h,{'Class1','Class2','Class1 Tr','Class2 Tr'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');

% Testing data points
figure('Name', 'K-nearest neighbor - 2D division feature curve testdata (from testData.mat)', 'NumberTitle', 'off')
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
hold on
h(3:4) = gscatter(featureMatrix_l(:,4),featureMatrix_l(:,5),Clte);
legend(h,{'Class1','Class2','Class1 Te','Class2 Te'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');

% ROC curve one vs one
[fpr,tpr,T,AUC,OPTROCPT] = perfcurve(Clte,score(:,1),1);
AUC
figure;
plot(fpr,tpr,'.-')
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by knn')

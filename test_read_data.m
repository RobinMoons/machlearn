close all;
clear all;
clc;

%Selected activity: DRINKING
%Explanation of variables
% featureMatrix_s : features from the whole small dataset
% featureMatrix_l : features from the whole large dataset
% featureMatrix_1_3 : features from the 1/3 dataset
% featureMatrix_2_3 : features from the 2/3 dataset
% Class_s : class from the whole small dataset (chosen DRINKING)
% Class_l : class from the whole large dataset (chosen DRINKING)
% Class_1_3 : class from the 1/3 dataset (chosen DRINKING)
% Class_2_3 : class from the 2/3 dataset (chosen DRINKING)

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
col1 = [drinkingFeature(:,4); shoeFeature(:,4); brushingFeature(:,4); writingFeature(:,4)];
col2 = [drinkingFeature(:,5); shoeFeature(:,5); brushingFeature(:,5); writingFeature(:,5)];
featureMatrix_s = [col1,col2];
%create class (used to check the results)
amountDrinking = numel(smallData.drinking);
amountBrush = numel(smallData.brush);
amountShoe = numel(smallData.shoe);
amountWriting = numel(smallData.writing);
Class_s = [ones(amountDrinking,1);2*ones(amountBrush + amountShoe + amountWriting,1)];   
%Scatter plots from features 
featureMatrix_training = [drinkingFeature;shoeFeature;brushingFeature;writingFeature];
figure, gplotmatrix(featureMatrix_training,[],Class_s);
title('gplotmatrix featureMatrix')

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
featureMatrix_1_3 = [featureMatrix_1_3(:,4),featureMatrix_1_3(:,5)];
%extract features from 2/3 set
featureMatrix_2_3 = featureExtraction(activities_2_3);
featureMatrix_2_3 = [featureMatrix_2_3(:,4),featureMatrix_2_3(:,5)];

%% Extract features from large data set
%segmentatie large dataset
numberSamples = numel(largeData.AthensTest_Accel_LN_X_CAL);
size = 2000;
numberActivities = floor(numberSamples / size);
drinkingActivityCounter = 0;
Class_l = [];
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
            drinkingCounter = drinkingCounter +1;
        end 
    end
    if (drinkingCounter > (size/2))
        drinkingActivityCounter = drinkingActivityCounter +1;
        Class_l = vertcat(Class_l,1);
    else
        Class_l = vertcat(Class_l,2);
    end
end
%extract features
featureMatrix_l = featureExtraction(testActiviteiten);
featureMatrix_l = [featureMatrix_l(:,4),featureMatrix_l(:,5)];
%create class (used to check the results)
%Class_l = [ones(drinkingActivityCounter,1);2*ones((numberActivities - drinkingActivityCounter),1)];    

%% Decission tree for Binary classification
% train with 2/3 dataset
model_2_3 = fitctree(featureMatrix_2_3, Class_2_3);
% test with 2/3 dataset
[Cpred,score_2_3,node] = resubPredict(model_2_3);
% check accuracy trainingsdata
C_tree_2_3_training = confusionmat(Class_2_3,Cpred)
Acc_tree_2_3_training = trace(C_tree_2_3_training)/sum(sum(C_tree_2_3_training))
% test model with 1/3 data
[Cpred,score_1_3] = predict(model_2_3,featureMatrix_1_3);
% check accuracy testdata
C_tree_1_3_test = confusionmat(Class_1_3,Cpred)
Acc_tree_1_3_test = trace(C_tree_1_3_test)/sum(sum(C_tree_1_3_test))

% train with small dataset
model_s = fitctree(featureMatrix_s, Class_s);
% test with small dataset
[Cpred,score_s,node] = resubPredict(model_s);
% check accuracy trainingsdata
C_tree_s_training = confusionmat(Class_s,Cpred)
Acc_tree_s_training = trace(C_tree_s_training)/sum(sum(C_tree_s_training))
% test with large dataset
[Cpred,score_l] = predict(model_s,featureMatrix_l);
% check accuracy testdata
C_tree_l_test = confusionmat(Class_l,Cpred)
Acc_tree_l_test = trace(C_tree_l_test)/sum(sum(C_tree_l_test))

% Visualisation of results
result = createGscatter('scatter plots decission tree',featureMatrix_2_3,Class_2_3,featureMatrix_1_3,Class_1_3,model_2_3,featureMatrix_s,Class_s,featureMatrix_l,Class_l,model_s);
% ROC curves and AUC's
[result, AUC_tree_2_3_tr, AUC_tree_1_3_te, AUC_tree_s_tr, AUC_tree_l_te] = createAUC('ROC curves decission tree',score_2_3,Class_2_3,score_1_3,Class_1_3,score_s,Class_s,score_l,Class_l);
AUC_tree_2_3_tr
AUC_tree_1_3_te
AUC_tree_s_tr
AUC_tree_l_te

%% SVM for binary classification
% train with 2/3 dataset
SVMModel_2_3 = fitcsvm(featureMatrix_2_3,Class_2_3);
% test with 2/3 dataset
[Cpred,score_2_3,node] = resubPredict(SVMModel_2_3);
% check accuracy trainingsdata
C_svm_2_3_training = confusionmat(Class_2_3,Cpred)
Acc_svm_2_3_training = trace(C_svm_2_3_training)/sum(sum(C_svm_2_3_training))
% test model with 1/3 data
[Cpred,score_1_3] = predict(SVMModel_2_3,featureMatrix_1_3);
% check accuracy testdata
C_svm_1_3_test = confusionmat(Class_1_3,Cpred)
Acc_svm_1_3_test = trace(C_svm_1_3_test)/sum(sum(C_svm_1_3_test))

% train with small dataset
SVMModel_s = fitcsvm(featureMatrix_s, Class_s);
% test with small dataset
[Cpred,score_s,node] = resubPredict(SVMModel_s);
% check accuracy trainingsdata
C_svm_s_training = confusionmat(Class_s,Cpred)
Acc_svm_s_training = trace(C_svm_s_training)/sum(sum(C_svm_s_training))
% test with large dataset
[Cpred,score_l] = predict(SVMModel_s,featureMatrix_l);
% check accuracy testdata
C_svm_l_test = confusionmat(Class_l,Cpred)
Acc_svm_l_test = trace(C_svm_l_test)/sum(sum(C_svm_l_test))

% Visualisation of results
result = createGscatter('scatter plots SVM',featureMatrix_2_3,Class_2_3,featureMatrix_1_3,Class_1_3,SVMModel_2_3,featureMatrix_s,Class_s,featureMatrix_l,Class_l,SVMModel_s)
% ROC curves and AUC's
[result, AUC_svm_2_3_tr, AUC_svm_1_3_te, AUC_svm_s_tr, AUC_svm_l_te] = createAUC('ROC curves SVM',score_2_3,Class_2_3,score_1_3,Class_1_3,score_s,Class_s,score_l,Class_l);
AUC_svm_2_3_tr
AUC_svm_1_3_te
AUC_svm_s_tr
AUC_svm_l_te

%% Naive Bayes for binary classification 
% train with 2/3 dataset
BayesModel_2_3 = fitcnb(featureMatrix_2_3,Class_2_3);
% test with 2/3 dataset
[Cpred,score_2_3,node] = resubPredict(BayesModel_2_3);
% check accuracy trainingsdata
C_bayes_2_3_training = confusionmat(Class_2_3,Cpred)
Acc_bayes_2_3_training = trace(C_bayes_2_3_training)/sum(sum(C_bayes_2_3_training))
% test model with 1/3 data
[Cpred,score_1_3] = predict(BayesModel_2_3,featureMatrix_1_3);
% check accuracy testdata
C_bayes_1_3_test = confusionmat(Class_1_3,Cpred)
Acc_bayes_1_3_test = trace(C_bayes_1_3_test)/sum(sum(C_bayes_1_3_test))

% train with small dataset
BayesModel_s = fitcnb(featureMatrix_s, Class_s);
% test with small dataset
[Cpred,score_s,node] = resubPredict(BayesModel_s);
% check accuracy trainingsdata
C_bayes_s_training = confusionmat(Class_s,Cpred)
Acc_bayes_s_training = trace(C_bayes_s_training)/sum(sum(C_bayes_s_training))
% test with large dataset
[Cpred,score_l] = predict(BayesModel_s,featureMatrix_l);
% check accuracy testdata
C_bayes_l_test = confusionmat(Class_l,Cpred)
Acc_bayes_l_test = trace(C_bayes_l_test)/sum(sum(C_bayes_l_test))

% Visualisation of results
result = createGscatter('scatter plots Bayes',featureMatrix_2_3,Class_2_3,featureMatrix_1_3,Class_1_3,BayesModel_2_3,featureMatrix_s,Class_s,featureMatrix_l,Class_l,BayesModel_s)
% ROC curves and AUC's
[result, AUC_bayes_2_3_tr, AUC_bayes_1_3_te, AUC_bayes_s_tr, AUC_bayes_l_te] = createAUC('ROC curves Bayes',score_2_3,Class_2_3,score_1_3,Class_1_3,score_s,Class_s,score_l,Class_l);
AUC_bayes_2_3_tr
AUC_bayes_1_3_te
AUC_bayes_s_tr
AUC_bayes_l_te

%% K-nearest neighbour for binary classification   %%uitbreiden met K en dist kijk naar dichtsbijzijnde, 3 dichtsbijzijnde of optimized
% train with 2/3 dataset
KnnModel_2_3 = fitcknn(featureMatrix_2_3,Class_2_3);
% use this line for the 3D graph
%KnnModel = fitcknn(featureMatrix_2_3,Class_2_3,'OptimizeHyperparameters','auto') 
% test with 2/3 dataset
[Cpred,score_2_3,node] = resubPredict(KnnModel_2_3);
% check accuracy trainingsdata
C_Kn_2_3_training = confusionmat(Class_2_3,Cpred)
Acc_Kn_2_3_training = trace(C_Kn_2_3_training)/sum(sum(C_Kn_2_3_training))
% test model with 1/3 data
[Cpred,score_1_3] = predict(KnnModel_2_3,featureMatrix_1_3);
% check accuracy testdata
C_Kn_1_3_test = confusionmat(Class_1_3,Cpred)
Acc_Kn_1_3_test = trace(C_Kn_1_3_test)/sum(sum(C_Kn_1_3_test))

% train with small dataset
KnnModel_s = fitcknn(featureMatrix_s, Class_s);
% use this line for the 3D graph
%KnnModel = fitcknn(featureMatrix_s, Class_s,'OptimizeHyperparameters','auto')
% test with small dataset
[Cpred,score_s,node] = resubPredict(KnnModel_s);
% check accuracy trainingsdata
C_Kn_s_training = confusionmat(Class_s,Cpred)
Acc_Kn_s_training = trace(C_Kn_s_training)/sum(sum(C_Kn_s_training))
% test with large dataset
[Cpred,score_l] = predict(KnnModel_s,featureMatrix_l);
% check accuracy testdata
C_Kn_l_test = confusionmat(Class_l,Cpred)
Acc_Kn_l_test = trace(C_Kn_l_test)/sum(sum(C_Kn_l_test))

% Visualisation of results
result = createGscatter('scatter plots K nearest',featureMatrix_2_3,Class_2_3,featureMatrix_1_3,Class_1_3,KnnModel_2_3,featureMatrix_s,Class_s,featureMatrix_l,Class_l,KnnModel_s)
% ROC curves and AUC's
[result, AUC_Kn_2_3_tr, AUC_Kn_1_3_te, AUC_Kn_s_tr, AUC_Kn_l_te] = createAUC('ROC curves K nearest', score_2_3,Class_2_3,score_1_3,Class_1_3,score_s,Class_s,score_l,Class_l);
AUC_Kn_2_3_tr
AUC_Kn_1_3_te
AUC_Kn_s_tr
AUC_Kn_l_te

%% knn Classifier k == 3 dist = standard euclidian
% train with 2/3 dataset
KnnModel_kn3_2_3 = fitcknn(featureMatrix_2_3,Class_2_3,'NumNeighbors',3,'Standardize',1);
% use this line for the 3D graph
%KnnModel = fitcknn(featureMatrix_2_3,Class_2_3,'OptimizeHyperparameters','auto') 
% test with 2/3 dataset
[Cpred,score_2_3,node] = resubPredict(KnnModel_kn3_2_3);
% check accuracy trainingsdata
C_Kn3_2_3_training = confusionmat(Class_2_3,Cpred)
Acc_Kn3_2_3_training = trace(C_Kn3_2_3_training)/sum(sum(C_Kn3_2_3_training))
% test model with 1/3 data
[Cpred,score_1_3] = predict(KnnModel_kn3_2_3,featureMatrix_1_3);
% check accuracy testdata
C_Kn3_1_3_test = confusionmat(Class_1_3,Cpred)
Acc_Kn3_1_3_test = trace(C_Kn3_1_3_test)/sum(sum(C_Kn3_1_3_test))

% train with small dataset
KnnModel_kn3_s = fitcknn(featureMatrix_s, Class_s,'NumNeighbors',3,'Standardize',1);
% use this line for the 3D graph
%KnnModel = fitcknn(featureMatrix_s, Class_s,'OptimizeHyperparameters','auto')
% test with small dataset
[Cpred,score_s,node] = resubPredict(KnnModel_kn3_s);
% check accuracy trainingsdata
C_Kn3_s_training = confusionmat(Class_s,Cpred)
Acc_Kn3_s_training = trace(C_Kn3_s_training)/sum(sum(C_Kn3_s_training))
% test with large dataset
[Cpred,score_l] = predict(KnnModel_kn3_s,featureMatrix_l);
% check accuracy testdata
C_Kn3_l_test = confusionmat(Class_l,Cpred)
Acc_Kn3_l_test = trace(C_Kn3_l_test)/sum(sum(C_Kn3_l_test))

% Visualisation of results
result = createGscatter('scatter plots K nearest',featureMatrix_2_3,Class_2_3,featureMatrix_1_3,Class_1_3,KnnModel_kn3_2_3,featureMatrix_s,Class_s,featureMatrix_l,Class_l,KnnModel_kn3_s)
% ROC curves and AUC's
[result, AUC_Kn3_2_3_tr, AUC_Kn3_1_3_te, AUC_Kn3_s_tr, AUC_Kn3_l_te] = createAUC('ROC curves K nearest', score_2_3,Class_2_3,score_1_3,Class_1_3,score_s,Class_s,score_l,Class_l);
AUC_Kn3_2_3_tr
AUC_Kn3_1_3_te
AUC_Kn3_s_tr
AUC_Kn3_l_te

%% knn Classifier k == opt dist = opt
% train with 2/3 dataset 
KnnModel_knO_2_3 = fitcknn(featureMatrix_2_3,Class_2_3,'OptimizeHyperparameters','auto');
% use this line for the 3D graph
%KnnModel = fitcknn(featureMatrix_2_3,Class_2_3,'OptimizeHyperparameters','auto') 
% test with 2/3 dataset
[Cpred,score_2_3,node] = resubPredict(KnnModel_knO_2_3);
% check accuracy trainingsdata
C_KnO_2_3_training = confusionmat(Class_2_3,Cpred)
Acc_KnO_2_3_training = trace(C_KnO_2_3_training)/sum(sum(C_KnO_2_3_training))
% test model with 1/3 data
[Cpred,score_1_3] = predict(KnnModel_knO_2_3,featureMatrix_1_3);
% check accuracy testdata
C_KnO_1_3_test = confusionmat(Class_1_3,Cpred)
Acc_KnO_1_3_test = trace(C_KnO_1_3_test)/sum(sum(C_KnO_1_3_test))

% train with small dataset
KnnModel_knO_s = fitcknn(featureMatrix_s, Class_s,'NumNeighbors',3,'Standardize',1);
% use this line for the 3D graph
%KnnModel = fitcknn(featureMatrix_s, Class_s,'OptimizeHyperparameters','auto')
% test with small dataset
[Cpred,score_s,node] = resubPredict(KnnModel_knO_s);
% check accuracy trainingsdata
C_KnO_s_training = confusionmat(Class_s,Cpred)
Acc_KnO_s_training = trace(C_KnO_s_training)/sum(sum(C_KnO_s_training))
% test with large dataset
[Cpred,score_l] = predict(KnnModel_knO_s,featureMatrix_l);
% check accuracy testdata
C_KnO_l_test = confusionmat(Class_l,Cpred)
Acc_KnO_l_test = trace(C_KnO_l_test)/sum(sum(C_KnO_l_test))

% Visualisation of results
result = createGscatter('scatter plots K nearest',featureMatrix_2_3,Class_2_3,featureMatrix_1_3,Class_1_3,KnnModel_knO_2_3,featureMatrix_s,Class_s,featureMatrix_l,Class_l,KnnModel_knO_s)
% ROC curves and AUC's
[result, AUC_KnO_2_3_tr, AUC_KnO_1_3_te, AUC_KnO_s_tr, AUC_KnO_l_te] = createAUC('ROC curves K nearest', score_2_3,Class_2_3,score_1_3,Class_1_3,score_s,Class_s,score_l,Class_l);
AUC_KnO_2_3_tr
AUC_KnO_1_3_te
AUC_KnO_s_tr
AUC_KnO_l_te

%% Bram: Testen van precision-recall curve.
% howto: https://www.quora.com/How-can-I-plot-a-precision-recall-curve-in-MATLAB
% ROC curve one vs one
% help resubPredict
[~,score] = resubPredict(BayesModel_s);
%Class1 vs Class2
%help perfcurve
[fpr,tpr,T,AUC_perf] = perfcurve(Class_s,score(:,1),1,'xCrit','reca','yCrit','prec');
AUC_perf
figure('Name', 'Precision-recall curve testdata (from testData.mat)', 'NumberTitle', 'off')
plot(fpr,tpr)
xlabel('Recal')
ylabel('Precision')
title('Precision-recall curve for Classification by Classification Trees')

% Also divide the feature space in the region of the positive instances and the region of the negative instances. 
% Visualise also in the feature space the training instances. 
% Plot the Receiver-Operating-Characteristic (ROC) and also calculate the area-under-the-curve (AUC).
% Use for this purpose the MATLAB instructions perfcurve. 
% The score is obtained by the MATLAB instruction resubPredict for trees. 
% Also calculate the confusion matrix on the training set. Use for that the instruction confusionmat in MATLAB.  
% Extract the accuracy of the binary classifier. 
% Note that calculating performance measures on the training data gives too optimistic results (overfitting). 



%% Naive bayes as Binary classificcation

% 
% % Classification
% BayesModel = fitcnb(featureMatrix_s,Class_s);
% 
% BayesModel
% BayesModel.DistributionParameters
% Params = cell2mat(BayesModel.DistributionParameters);
% 
% Mu = Params([1 3],1:2); % Extract the means
% Sigma = zeros(2,2,3);
% 
% face = {'r' , 'b' }
% 
% for j = 1:2
%     Sigma(:,:,j) = diag(Params(2*j,:)).^2; % Create diagonal covariance matrix
%     
%     surf(x1Grid,x2Grid,reshape(mvnpdf(xGrid,Mu(j,:),Sigma(:,:,j)),size(x1Grid)),'FaceAlpha',0.5,'FaceColor',face{j})
%         % Draw contours for the multivariate normal distributions
% end
% 


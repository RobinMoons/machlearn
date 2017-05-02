%% Machine learning example
clc
clear
close all


%% Make simulated data

%Make 4 classes of Gaussian distributed data
%First class mean in origin and sigma 1
%Multi variate normal random 
%help mvnrnd

%Class 1
n = 100;
sigma = .1;
SigmaInd = sigma .* [1 0.2 ; 0.2 1 ];
X1 = mvnrnd([0 0 ], SigmaInd, n);

%Class 2
X2 = mvnrnd([1 0 ], SigmaInd, n/4);

%Class 2
X3 = mvnrnd([0 1 ], 2*SigmaInd, n/4);

%Class 2
X4 = mvnrnd([-1 0 ], SigmaInd, n/2);

% sigma = .1;
% SigmaInd = sigma .* [1 0.8 ; 0.8 1 ];
% X1 = mvnrnd([0 0 ], SigmaInd, n);
% 
% %Class 2
% X2 = mvnrnd([0.5 0 ], SigmaInd, n/4);
% 
% %Class 2
% X3 = mvnrnd([0 1 ], SigmaInd, n/4);
% 
% %Class 2
% X4 = mvnrnd([-0.5 0 ], SigmaInd, n/2);


X = [X1;X2;X3;X4];
Class = [ones(n,1);2*ones(n,1)];

%% Visualisation of simulated data

%help gplotmatrix
gplotmatrix(X,[],Class)

%% Binary classification training and test set
%Class1 vs class2
figure 
%help gscatter
gscatter(X(1:2*n,1),X(1:2*n,2),Class(1:2*n))
%Establish a training and test set
%help randperm

p = randperm(2*n);
%Training set 50% of data
Xtr = X(p(1:n),:);
Cltr = Class(p(1:n));
figure
title('Training set')
gscatter(Xtr(:,1),Xtr(:,2),Cltr)
hold on

%Test set other 50% of data

Xte = X(p(n+1:2*n),:);
Clte = Class(p(n+1:2*n));

%help meshgrid
d = 0.1;
[x1Grid,x2Grid] = meshgrid(min(Xtr(:,1)):d:max(Xtr(:,1)),...
    min(Xtr(:,2)):d:max(Xtr(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];



%% Naive Bayes Classifier

%help fitcnb
Mdl = fitcnb(Xtr,Cltr);

Mdl
Mdl.DistributionParameters
Params = cell2mat(Mdl.DistributionParameters);

Mu = Params([1 3],1:2); % Extract the means
Sigma = zeros(2,2,3);

face = {'r' , 'b' }

for j = 1:2
    Sigma(:,:,j) = diag(Params(2*j,:)).^2; % Create diagonal covariance matrix
    
    surf(x1Grid,x2Grid,reshape(mvnpdf(xGrid,Mu(j,:),Sigma(:,:,j)),size(x1Grid)),'FaceAlpha',0.5,'FaceColor',face{j})
        % Draw contours for the multivariate normal distributions
end


%% Accuracy on trainings data
%help resubPredict
[Cpred_tr,score,node] = resubPredict(Mdl);
%help confusionmat
C = confusionmat(Cltr,Cpred_tr)
accuracy = trace(C)/sum(sum(C))





%% Accuracy on test data
%help confusionmat
[Cpred,score] = predict(Mdl,Xte(:,1:2));
C = confusionmat(Clte,Cpred)
accuracy = trace(C)/sum(sum(C))


%% visualisation of results

%help meshgrid
d = 0.01;
[x1Grid,x2Grid] = meshgrid(min(Xtr(:,1)):d:max(Xtr(:,1)),...
    min(Xtr(:,2)):d:max(Xtr(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

labels = predict(Mdl,xGrid);

% Training data points
figure
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
hold on
h(3:4) = gscatter(Xtr(:,1),Xtr(:,2),Cltr);
legend(h,{'Class1','Class2','Class1 Tr','Class2 Tr'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');

% Testing data points
figure 

h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
hold on
h(3:4) = gscatter(Xte(:,1),Xte(:,2),Clte);
legend(h,{'Class1','Class2','Class1 Te','Class2 Te'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');


%% ROC curve one vs one

%help perfcurve
[fpr,tpr,T,AUC,OPTROCPT] = perfcurve(Clte,score(:,1),1);
AUC
figROC=figure;
plot(fpr,tpr,'.-')
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by Classification SVM linear')


%% Model likelihood functions with GMM

%help meshgrid
d = 0.1;
[x1Grid,x2Grid] = meshgrid(min(Xtr(:,1)):d:max(Xtr(:,1)),...
    min(Xtr(:,2)):d:max(Xtr(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

% take class 1 and build a Gausian Mixture Model
index = find (Cltr == 1);
GMModelCl1 = fitgmdist(Xtr(index,:),1);
figure
title('Training set')
gscatter(Xtr(:,1),Xtr(:,2),Cltr)
hold on
surf(x1Grid,x2Grid,reshape(pdf(GMModelCl1,xGrid),size(x1Grid)),'FaceAlpha',0.5,'FaceColor','r')


% take class 2 and build a Gausian Mixture Model
index = find (Cltr == 2);
GMModelCl2 = fitgmdist(Xtr(index,:),2);

surf(x1Grid,x2Grid,reshape(pdf(GMModelCl2,xGrid),size(x1Grid)),'FaceAlpha',0.5,'FaceColor','b')

% When is class 1 detected
labels1 = double(pdf(GMModelCl1,Xte)>=pdf(GMModelCl2,Xte));
% When is class 2 detected
labels2 = 2*double(pdf(GMModelCl2,Xte)>pdf(GMModelCl1,Xte));

label = labels1 + labels2;
% scores are the posteriors when looking at a balance dataset
Score = pdf(GMModelCl1,Xte)./(pdf(GMModelCl1,Xte)+pdf(GMModelCl2,Xte));

%% Accuracy on test data
%help confusionmat

C = confusionmat(Clte,label)
accuracy = trace(C)/sum(sum(C))


%% ROC curve one vs one

%help perfcurve
[fpr,tpr,T,AUC,OPTROCPT] = perfcurve(Clte,Score,1);
AUC
figure(figROC);
plot(fpr,tpr,'.-')
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by Classification ')







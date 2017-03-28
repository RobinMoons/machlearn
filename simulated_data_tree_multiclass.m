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
SigmaInd = sigma .* [1 0 ; 0 1  ];
X1 = mvnrnd([0 0], SigmaInd, n);

%Class 2
X2 = mvnrnd([1 0 ], SigmaInd, n);

%Class 3
X3 = mvnrnd([0 1 ], SigmaInd, n);

%Class 4
X4 = mvnrnd([0 -1 ], SigmaInd, n);

X = [X1;X2;X3;X4];
Class = [ones(n,1);2*ones(n,1);3*ones(n,1);4*ones(n,1)];

%% Visualisation of simulated data

%help gplotmatrix
gplotmatrix(X,[],Class)

%% Establish a training and test set
%help randperm

p = randperm(4*n);
%Training set 50% of data
Xtr = X(p(1:2*n),:);
Cltr = Class(p(1:2*n));
figure
title('Training set')
gscatter(Xtr(:,1),Xtr(:,2),Cltr)


%Test set other 50% of data

Xte = X(p(2*n+1:4*n),:);
Clte = Class(p(2*n+1:4*n));


%% Training a decission tree

%help fitctree
Mdl = fitctree(Xtr(:,1:2),Cltr);
view(Mdl)
view(Mdl,'Mode','graph')
%% Accuracy on trainings data
%help resubPredict
[Cpred_tr,score,node] = resubPredict(Mdl);
%help confusionmat
% Each row is a class,
C = confusionmat(Cltr,Cpred_tr)
accuracy = trace(C)/sum(sum(C))


%% Test a decission tree
%help predict
Cpred = predict(Mdl,Xte(:,1:2));


%% Accuracy on test data
%help confusionmat
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
h(1:4) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5; 0.5 0.5 0.1; 0.1 0.5 0.1 ]);
hold on
h(5:8) = gscatter(Xtr(:,1),Xtr(:,2),Cltr);
legend(h,{'Class1','Class2','Class3','Class4','Class1 Tr','Class2 Tr','Class3 Tr','Class4 Tr'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');

% Testing data points
figure 

h(1:4) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5; 0.5 0.5 0.1; 0.1 0.5 0.1 ]);
hold on
h(5:8) = gscatter(Xte(:,1),Xte(:,2),Clte);
legend(h,{'Class1','Class2','Class3','Class4','Class1 Te','Class2 Te','Class3 Te','Class4 Te'},...
   'Location','Northwest');
xlabel('x1');
ylabel('x2');

%% ROC curve one vs one
% help resubPredict
% [~,score] = resubPredict(Mdl);
% Class1 vs Class2
%help perfcurve
[fpr,tpr,T,AUC,OPTROCPT] = perfcurve(Cltr,score(:,1),1);
AUC
figure
plot(fpr,tpr,'.-')
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by Classification Trees')
hold off






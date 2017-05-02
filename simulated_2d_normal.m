%% Machine learning example
clc
clear
close all


%% Make simulated data


%Class 1
n = 100;
sigma = .1;
SigmaInd = sigma .* [1 .9 ; .9 4 ];
Mu = [1 1]
X = mvnrnd(Mu, SigmaInd, n);

%% plot datapoints
figure
title('data points')
gscatter(X(:,1),X(:,2))
xlabel('x1')
ylabel('x2')
axis equal
hold on

%help meshgrid
d = 0.1;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

surf(x1Grid,x2Grid,reshape(mvnpdf(xGrid,Mu,SigmaInd),size(x1Grid)),'FaceAlpha',0.5,'FaceColor','r')

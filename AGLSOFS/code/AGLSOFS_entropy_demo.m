clear,clc,warning('off');
method = "AGLSOFS_norm";
load("Yale"); %load data
X = mapminmax(X',0,1)';
X1 = X;
Y1 = Y;
Y = full(ind2vec(Y'))';
[~,c] = size(Y);
[XL,YL,XU,YU] = split_train_test(X1,Y1,c,0.3);
X = [XL;XU];
YL = full(ind2vec(YL'))';

%set parameters
param.beta = 1;
param.alpha = 1;
param.la1 = 1;
param.la2 = 1;
param.gama = 0.5;

%train to get W and loss
[W,objs] = AGLSOFS_entropy(X,YL,param);
plot(objs(2:end));

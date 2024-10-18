function [W,objs] = AGLSOFS_entropy(X,YL,param)
% Ref: "Adaptive orthogonal semi-supervised feature selection with reliable label matrix learning"
% ipm, 2024. 
% Authors: Huming Liao, Hongmei Chen, Tengyu Yin,Chuan Luo, Shi-Jinn Horng,Tianrui Li.

% Input
% X: n*d data matrix
% YL: nl*c label matrix
% param: parameters

%Output
% W: d*c projection matrix
% obj: loss value.

beta = param.beta;
alpha = param.alpha;
la1 = param.la1;
la2 = param.la2;
gama = param.gama;
max_ite = 200;

[n,d] = size(X);
[nl,c] = size(YL);
YU = zeros(n-nl,c);
Y = [YL;YU];
W = rand(d,c);
Dw = eye(d);
S = zeros(n,n);

for it=1:max_ite
    
    %1.update YU
    T = X*W./alpha;
    for ji = nl+1:n
        Y(ji,:) = EProjSimplex_new(T(ji,:));
    end
    
    %2.update S
    dist = gama.*pdist2(X*W,X*W,'squaredeuclidean') + (1-gama).*pdist2(X,X,'squaredeuclidean');
    dd = exp((-1.*dist)./la2);
    for ji = 1:n
        S(ji,:) = dd(ji,:) / sum(dd(ji,:));
    end
    S = (S+S')/2;
    L = diag(sum(S,2)) - S; 
    
    % calculate the loss
    o1 = trace((X*W-alpha.*Y)'*(X*W-alpha.*Y)) + la1*trace(W'*Dw*W);
    SS = S + eps;
    o2 = 2*beta*gama*trace(W'*X'*L*X*W)+2*beta*(1-gama)*trace(X'*L*X)+beta*la2*sum(SS.*log(SS),'all');
    objs(it) = o1 + o2;
    if it > 1 && abs(objs(it-1)-objs(it)) < 0.000001
        break;
    end
    
    %3.update P
    J = X'*X + 2*la1.*Dw + (2*beta*gama).*X'*L*X;
    M = alpha.*X'*Y;
    
    [W] = update_P(J,M,W,1);
    Dw = diag(0.5 * (sqrt(sum(W.^2,2)+eps)).^(-1));
end
end
    


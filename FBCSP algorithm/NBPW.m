function [pwx] = NBPW(Xtrain, class, xeval, w)
% NBPW Naive Bayesian Parzen Window classifier
%
%   INPUT:
%   'Xtrain' is the training features matrix;
%   'class' is array with classes associated to each row of 'Xtrain';
%   'xeval' is the array of features of a signal to classify;
%   'w' is the class of interest.
%
%   OUTPUT:
%   'pwx' is the probability that the signal to classify belong to the
%   class 'w'
%
%
%  authors:         A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2020/05/12

    % probability of 'xeval' given 'w = 1,2' with naive assumption
    d = length(xeval);      % number of selected features
    kw1 = find(class == w);
    kw2 = find(class ~= w);
    nw1 = length(kw1);      % number of elements in class 'w'
    nw2 = length(kw2);      % number of elements not in class 'w'
    nt = size(Xtrain,1);
    pxw1 = 1;
    pxw2 = 1;
    for j = 1:d
       sig = std(xeval(j) - Xtrain(:,j));
       h = sig*(4/(3*nt))^(1/5);
       phi = exp(-((xeval(j) - Xtrain(:,j)).^2)/(2*h^2))/sqrt(2*pi*h^2);
       pxw1 = pxw1*sum(phi(kw1))/nw1; 
       pxw2 = pxw2*sum(phi(kw2))/nw2;
    end

    % prior class probability
    Pw1 = nw1/nt;
    Pw2 = nw2/nt;
    
    % probability of 'xeval'
    px = pxw1*Pw1 + pxw2*Pw2;
    
    % probability of class 'w' given 'xeval'
    pwx = pxw1*Pw1/px;
end


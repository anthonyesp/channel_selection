% This script executes training of the FBCSP algorithm with Bayesian
% classifier, and then evaluation. It is meant to have reference
% classification accuracies when all channels are exploited.
%
%
% This is related to the analyses published in
%  ** Arpaia, P., Donnarumma, F., Esposito, A. and Parvis, M., 2021. 
%    "Channel selection for optimal EEG measurement in motor imagery-based 
%    brain-computer interfaces." International Journal of Neural Systems, 
%    31(03), p.2150003, doi: 10.1142/S0129065721500039
%
%  author:          A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2023/01/03
%

close all;
clear;
clc

% Init classes of interest
class1 = 1;
class2 = 2;

% Algorithm hyperparameters
m = 2;                      % CSP components
k = 5;                      % MIBIF components
ch = 1:22;                  % EEG channels

% Data selection
subj = 'A01';               % one subject per time
runs = 4:9;                 % note: for A04T ONLY, use "runs = 2:7"
trials = 1:48;

% Time window of the signal
tmin = 3;
tmax = 6;

%% TRAINING
% Signals extraction
[imagery, classes, f_samp] = extraction(strcat(subj,'T.mat'), runs, ch, trials, tmin, tmax);
class   = classes(classes==class1|classes==class2);
imagery = imagery(classes==class1|classes==class2,:);

% FILTER BANK
eeg = filterBank(imagery,f_samp);

% SPATIAL FILTERING
[W1, W2] = CSPtrain(eeg, class, ch, m);
[V1, V2, ~, ~, Y] = CSPapply(eeg, class, W1, W2, [], []);

% FEATURES SELECTION
% Indexes with maximum Mutual-Information and their complementary
I1 = MIBIF(V1,Y,class1,m,k);
I2 = MIBIF(V2,Y,class2,m,k);

% Selection
f1 = V1(:,I1);
f2 = V2(:,I2);

% CLASSIFIER TRAINING
% Composite vector (Naive Bayesian training)
f = [f1, f2];
cl = Y;

%% EVALUATION
% Signals extraction
[imagery, classes, f_samp] = extraction(strcat(subj,'E.mat'), runs, ch, trials, tmin, tmax);
class = classes(classes==class1|classes==class2);
imagery = imagery(classes==class1|classes==class2,:);

% FILTER BANK
eeg = filterBank(imagery,f_samp);

% SPATIAL FILTERING
[V1, V2, ~, ~, Y] = CSPapply(eeg, class, W1, W2, [], []);

% FEATURES SELECTION
% Selection
feval1 = V1(:,I1);
feval2 = V2(:,I2);

% CLASSIFICATION
% Composite vector
feval = [feval1, feval2];
cleval = Y;

% Naive Bayesian Parzen Window classifier
nt = length(cleval);
proba = zeros(nt,1);
index = zeros(nt,1);
ind = index;
for i = 1:nt
    pwx1 = NBPW(f1, cl, feval1(i,:), class1);
    pwx2 = NBPW(f2, cl, feval2(i,:), class2);
    [proba(i), index(i)] = max([pwx1 pwx2]);
end
ind(index==1) = class1;
ind(index==2) = class2;

% Classification accuracy
accuracy = (sum(ind == cleval)/nt)*100;
 
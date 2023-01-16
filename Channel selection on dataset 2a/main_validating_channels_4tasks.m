% This main script validates the found best channels to classify four
% tasks of motor imagery by considering the classification performance in
% training and evaluation with data from different sessions (TE).
%
% The dataset 2a of BCI Competition IV (2008) is here considered. The
% possible six pairs of classes are taken into account
%
% The procedure is described in details in the paper "Channel selection for
% optimal EEG measurement in motor imagery-based Brain-Computer Interfaces"
%
%  authors:         A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2023/01/03

close all;
clear;
clc

%% ALGORITHM PARAMETERS
% Number of subject taken into account for the competition:
n_subjects = 9;

% % Algorithm hyperparameters
% m = 2;                      % CSP components
% k = 5;                      % MIBIF components

% Data selection
chmax = 22;
ch = 1:chmax;
runs = 4:9; 
trials = 1:48;

% Time window of the signal
tmin = 3;
tmax = 6;

%% LOADING
% Subject data
tic
subjects_data = cell(1,n_subjects);
for s = 1:n_subjects
    subj = strcat('A0',num2str(s));
    
    % correcting the runs for A04T
    if (s == 4)
       runs = 2:7;
    end

    % Training data
    [imagery, classes, f_samp]  = extraction(strcat(subj,'T.mat'),runs,ch,trials,tmin,tmax);
    
    if (exist('hd','var'))
        EEG_T = filterBank2(imagery,f_samp,hd);
    else
        [EEG_T, hd] = filterBank2(imagery,f_samp,[]); 
    end
    CLASS_T = classes;
    
    % reshape to have channels as first dimension
    EEG_T = reshape(EEG_T, [chmax, size(EEG_T,1)/chmax, size(EEG_T,2), size(EEG_T,3)]);
    CLASS_T = reshape(CLASS_T, [chmax, size(CLASS_T,1)/chmax]);
    
    % Evaluation data
    runs = 4:9;                     % re-fixing the correct runs to extract
    [imagery, classes, f_samp]  = extraction(strcat(subj,'E.mat'),runs,ch,trials,tmin,tmax);
    
    if (exist('hd','var'))
        EEG_E = filterBank2(imagery,f_samp,hd);
    else
        [EEG_E, hd] = filterBank2(imagery,f_samp,[]); 
    end
    CLASS_E = classes;
    
    % reshape to have channels as first dimension
    EEG_E = reshape(EEG_E, [chmax, size(EEG_E,1)/chmax, size(EEG_E,2), size(EEG_E,3)]);
    CLASS_E = reshape(CLASS_E, [chmax, size(CLASS_E,1)/chmax]);
    
    % Organize data in a struct
    subjects_data(s) = {struct('EEG_T',EEG_T,'CLASS_T',CLASS_T,'EEG_E',EEG_E,'CLASS_E',CLASS_E)};
end
toc             % about 14 min from tic

%% TRAINING AND EVALUATION
tic

% Accuracy and Standard Deviation matrix
ACC = zeros(n_subjects, chmax);
STD = zeros(n_subjects, chmax);

% Subject iterations
for s = 1:n_subjects

    % Subject Selection
    subj = strcat('A0',num2str(s));
    string = strcat('current subject: A0', num2str(s));
    disp(string);

    % currently, A0xT and A0xE are inverted
    EEG_T = subjects_data{1,s}.EEG_E;
    CLASS_T = subjects_data{1,s}.CLASS_E;
    
    EEG_E = subjects_data{1,s}.EEG_T;
    CLASS_E = subjects_data{1,s}.CLASS_T;
    
    % Iteration at increasing channel number
    CH = [2,16,12,8,14,6,17,10,22,19,7,21,18,4,20,15,13,9,11,5,3,1];

    % channel iteration
    for iteration = 1:chmax
        string = strcat('    current iteration: ', num2str(iteration));
        disp(string);

        % channels to consider
        ch = CH(1:iteration);
        
        % Select channels
        classT = CLASS_T(ch,:);
        eegT = EEG_T(ch,:,:,:);
        
        classE = CLASS_E(ch,:);
        eegE = EEG_E(ch,:,:,:);

        % Reshape
        nch = length(ch);
        classT = reshape(classT,[nch*size(classT,2), 1]);
        eegT = reshape(eegT,[nch*size(eegT,2), size(eegT,3), size(eegT,4)]);
        
        classE = reshape(classE,[nch*size(classE,2), 1]);
        eegE = reshape(eegE,[nch*size(eegE,2), size(eegE,3), size(eegE,4)]);

        ACC(s, iteration) = BCIalgorithFromCSP_4task(classT,eegT,classE,eegE,ch);
    end
    save(strcat('results_TE_4tasks_subj',num2str(s),'.mat'),'ACC','STD')
end

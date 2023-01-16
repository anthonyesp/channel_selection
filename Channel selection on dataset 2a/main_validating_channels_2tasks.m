% This main script validates the found best channels to classify two
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

% Initialize classes pairs
class_1 = [1, 1, 1, 2, 2, 3];
class_2 = [2, 3, 4, 3, 4, 4];
n_class_combination = length(class_1);

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

chmax = length(ch);

% Accuracy matrix
acc_all = zeros(n_subjects, chmax, n_class_combination);

for s = 1:n_subjects
    % TRAINING
    subj = strcat('A0',num2str(s));
    string = strcat('current subject: A0', num2str(s));
    disp(string);
    
    % currently, A0xT and A0xE are inverted!
    EEG_T = subjects_data{1,s}.EEG_E;
    CLASS_T = subjects_data{1,s}.CLASS_E;
   
    EEG_E = subjects_data{1,s}.EEG_T;
    CLASS_E = subjects_data{1,s}.CLASS_T;
	
	% DATA SELECTION
    for x_cl = 1:n_class_combination
        
        % Pair of tasks (2 tasks each iteration)
        class1 = class_1(x_cl);
        class2 = class_2(x_cl);
        
        string = strcat('  class_', num2str(class1),' vs class_', num2str(class2));
        disp(string);
		
        % Iteration at increasing channel number
        if (x_cl == 1)              % LHvsRH
            CH = [14, 16, 12, 8, 9, 11, 20, 5, 3, 4, 10, 17, 21, 15, 22, 18, 16, 19, 2, 7, 1, 13];
        elseif (x_cl == 2)          % LHvsF
            CH = [16, 12, 1, 17, 14, 6, 11, 22, 10, 18, 21, 5, 8, 9, 20, 19, 13, 15, 3, 2, 4, 7];
        elseif (x_cl == 3)          % LHvsT
            CH = [9, 12, 6, 17, 20, 19, 10, 4, 16, 13, 14, 18, 11, 15, 5, 3, 7, 8, 21, 2, 1, 22];
        elseif (x_cl == 4)          % RHvsF
            CH = [3, 14, 8, 9, 15, 17, 16, 20, 22, 18, 10, 1, 2, 11, 5, 21, 6, 4, 12, 13, 19, 7];
        elseif (x_cl == 5)          % RHvsT
            CH = [18, 8, 14, 13, 5, 16, 3, 17, 1, 19, 9, 11, 22, 12, 15, 7, 10, 20, 2, 21, 6, 4]; 
        elseif (x_cl == 6)          % FvT
            CH = [1, 10, 8, 16, 2, 22, 4, 5, 18, 20, 17, 11, 3, 21, 13, 14, 15, 9, 6, 19, 12, 7];
        else
            error('Unknown classes pair!');
        end
		
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

            % Select classes
            eegT = eegT(classT==class1|classT==class2,:,:);
            classT = classT(classT==class1|classT==class2);

            eegE = eegE(classE==class1|classE==class2,:,:);
            classE = classE(classE==class1|classE==class2);

            % accuracy calculation
            accuracy = BCIalgorithFromCSP_2task(classT,eegT,classE,eegE,ch);
            acc_all(s,iteration,x_cl) = accuracy;
        end
    end
    
    % saving data after each subject
    data = {struct('accuracy',acc_all)};
    save(strcat('results_TEinverted_2tasks_subj',num2str(s),'.mat'),'data')
end

toc             % about 26 min from tic
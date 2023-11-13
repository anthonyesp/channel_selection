% This main script progressively finds the best channels to classify two
% tasks of motor imagery by considering the classification performance in
% a training/evaluation procedure (TE). Hence, the model for discriminating
% motor imagery tasks is firstly trained and then evaluated.
%
% The dataset 3a of BCI Competition III (2006) is here considered. The
% possible six pairs of classes are taken into account
%
% The procedure is described in details in the paper "Channel selection for
% optimal EEG measurement in motor imagery-based Brain-Computer Interfaces"
% where the procedure is actually applied to the dataset 2a of BCI
% Competion IV (2008), but also in "Passive and active brain-computer 
% interfaces for rehabilitation in health 4.0. Measurement: Sensors, 18, 
% p.100246, doi: 10.1016/j.measen.2021.100246" for the dataset 3a case.
%
%  authors:         A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2023/01/03

close all; 
clear;
clc

%% ALGORITHM PARAMETERS
% Number of subject taken into account for the competition:
n_subjects = 3;

% Number of folds for Repeated Stratified Group K-Fold CV
k_folds = 6;

% Initialize classes pairs
class_1 = [1, 1, 1, 2, 2, 3];
class_2 = [2, 3, 4, 3, 4, 4];
n_class_combination = length(class_1);

% % Algorithm hyperparameters
% m = 2;                      % CSP components
% k = 5;                      % MIBIF components

% Data selection
subjects = ['k3b';'k6b';'l1b'];
chmax = 60;
ch = 1:chmax;
runs = 1:6;     % 6 runs for k6b e l1b, 9 runs per k3b.  
trials = 1:20;  % 40 trial - 20 di training e 20 di test.  

% Time window of the signal
tmin = 4;
tmax = 7;

%% LOADING
% Subject data
tic
subjects_data = cell(1,n_subjects);
for s = 1:n_subjects
    subj = subjects(s,:);
    pathTL = strcat(subj,'_truelabels.txt'); 
    
    data = divideTrainingTest(strcat(subj,'.mat'),pathTL);
    
    % Training data
    [imagery, classes, f_samp]  = extractionTraining(data, runs, ch, trials, tmin, tmax);
    
    if (exist('hd','var'))
        EEG_T = filterBank(imagery,f_samp,hd);
    else
        [EEG_T, hd] = filterBank(imagery,f_samp,[]); 
    end
    CLASS_T = classes;
    
    % reshape to have channels as first dimension
    EEG_T = reshape(EEG_T, [chmax, size(EEG_T,1)/chmax, size(EEG_T,2), size(EEG_T,3)]);
    CLASS_T = reshape(CLASS_T, [chmax, size(CLASS_T,1)/chmax]);
    
    % Evaluation data
    [imagery, classes, f_samp]  = extractionEvaluation(data, runs, ch, trials, tmin, tmax);
    
    if (exist('hd','var'))
        EEG_E = filterBank(imagery,f_samp,hd);
    else
        [EEG_E, hd] = filterBank(imagery,f_samp,[]); 
    end
    CLASS_E = classes;
        
    % reshape to have channels as first dimension
    EEG_E = reshape(EEG_E, [chmax, size(EEG_E,1)/chmax, size(EEG_E,2), size(EEG_E,3)]);
    CLASS_E = reshape(CLASS_E, [chmax, size(CLASS_E,1)/chmax]);
    
    % Organize data in a struct
    subjects_data(s) = {struct('EEG_T',EEG_T,'CLASS_T',CLASS_T,'EEG_E',EEG_E,'CLASS_E',CLASS_E)};
end

% Channel data
if (exist('results_channels_2tasks_60.mat','file') == 2)
    data = load('results_channels_2tasks_60.mat');
    ch_seq = data.data{1,1}.channel;
else
    error('Results of 60 channels do not exist!\nAlgorithm cannot continue...');
end
toc

%% TRAINING AND EVALUATION
results = zeros(n_subjects+3,chmax,n_class_combination);
results(1,:,:) = ch_seq;

for s = 1:n_subjects
    % Subject Selection
    string = strcat('current subject: ', subjects(s,:));
    disp(string);
	
    % Data selection
    EEG_T = subjects_data{1,s}.EEG_T;
    CLASS_T = subjects_data{1,s}.CLASS_T;
    EEG_E = subjects_data{1,s}.EEG_E;
    CLASS_E = subjects_data{1,s}.CLASS_E;
    
    % algorithm per channel number and per class
    for x_cl = 1:n_class_combination
        % Pair of tasks
        class1 = class_1(x_cl);
        class2 = class_2(x_cl);
        
        % Indexes of classes of interest
        CT = find(CLASS_T(1,:) == class1 | CLASS_T(1,:) == class2);
        CE = find(CLASS_E(1,:) == class1 | CLASS_E(1,:) == class2);
        
        string = strcat('  class_', num2str(class1),' vs class_', num2str(class2));
        disp(string);
        
        for ch_selected = 1:chmax
            % channel extraction
            ch = ch_seq(1:ch_selected,x_cl);
            if (mod(ch_selected,10) == 0)
                string = strcat('    current iteration: ', num2str(ch_selected));
                disp(string);
            end
            
            % extract train and test data
            classTr = CLASS_T(ch,CT);
            classEv = CLASS_E(ch,CE);
            
            eegTr = EEG_T(ch,CT,:,:);
            eegEv = EEG_E(ch,CE,:,:);
            
            % Reshape
            classTr = reshape(classTr,[ch_selected*size(classTr,2), 1]);
            eegTr = reshape(eegTr,[ch_selected*size(eegTr,2), size(eegTr,3), size(eegTr,4)]);

            classEv = reshape(classEv,[ch_selected*size(classEv,2), 1]);
            eegEv = reshape(eegEv,[ch_selected*size(eegEv,2), size(eegEv,3), size(eegEv,4)]);

            % Algorithm
            results(s+1,ch_selected,x_cl) = BCIalgorithFromCSP_2task(classTr,eegTr,classEv,eegEv,ch); 
        end        
    end
end

% mean and standard deviation
results(5,:,:) = mean(results(2:4,:,:),1);
results(6,:,:) = std(results(2:4,:,:),0,1);

% save results
save('results_TE_2task.mat','results');

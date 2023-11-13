% This main script progressively finds the best channels to classify two
% tasks of motor imagery by considering the classification performance in
% a 6-fold cross-validation (CV).
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
toc

%% CROSS-VALIDATION
channel_add = zeros(chmax,n_class_combination);
results = zeros(n_subjects+3,chmax,chmax,n_class_combination);

for ch_selected = 1:chmax
    string = strcat('current channel selected: ', num2str(ch_selected));
    disp(string);
    
    % Accuracy and Standard Deviation matrix
    ACC = zeros(n_subjects, chmax, n_class_combination);
    STD = zeros(n_subjects, chmax, n_class_combination);
    
    for s = 1:n_subjects

        % Subject Selection
        string = strcat('current subject: ', subjects(s,:));
        disp(string);

        EEG_T = subjects_data{1,s}.EEG_T;
        CLASS_T = subjects_data{1,s}.CLASS_T;

        % DATA SELECTION
        for x_cl = 1:n_class_combination

            % Pair of tasks (2 tasks each iteration)
            class1 = class_1(x_cl);
            class2 = class_2(x_cl);

            string = strcat('  class_', num2str(class1),' vs class_', num2str(class2));
            disp(string);

            % Iteration at increasing channel number
            if ch_selected == 1
                CH = [];
            else
                CH = channel_add(1:ch_selected-1,x_cl)';
            end

            % channel iteration
            for iteration = 1:chmax
                if (mod(iteration,10) == 0)
                    string = strcat('    current iteration: ', num2str(iteration));
                    disp(string);
                end

                % channels to consider
                ch = [CH, iteration];
                if (numel(ch) == numel(unique(ch)))

                    % Select channels
                    classT = CLASS_T(ch,:);
                    eegT = EEG_T(ch,:,:,:);
                    
                    % Select classes
                    eegT = eegT(:,classT(1,:) == class1|classT(1,:) == class2,:,:);
                    classT = classT(:,classT(1,:) == class1|classT(1,:) == class2);
                    
                    % stratified k-fold partition (no repeation)
                    cvPart.numTestSet = k_folds;
                    
                    ind1 = find(classT(1,:) == class1);
                    ind2 = find(classT(1,:) == class2);
                    
                    c1 = length(ind1)/k_folds;
                    c2 = length(ind2)/k_folds;
                    
                    for ind = 1:k_folds
                        cvPart.testInd{ind} = sort([ind1(floor((ind-1)*c1)+1:floor(ind*c1)) ind2(floor((ind-1)*c2)+1:floor(ind*c2))]);
                        temp = sort([ind1, ind2]);
                        temp(cvPart.testInd{ind}) = [];
                        cvPart.trainInd{ind} = temp;
                    end

                    nch = length(ch);
                    accuracy = zeros(1,cvPart.numTestSet);
                    for k = 1:cvPart.numTestSet
                        % Split dataset for train and test
                        classEv = classT(:,cvPart.testInd{k});
                        classTr = classT(:,cvPart.trainInd{k});

                        eegEv = eegT(:,cvPart.testInd{k},:,:);
                        eegTr = eegT(:,cvPart.trainInd{k},:,:);
                        
                         % Reshape
                        classTr = reshape(classTr,[nch*size(classTr,2), 1]);
                        eegTr = reshape(eegTr,[nch*size(eegTr,2), size(eegTr,3), size(eegTr,4)]);
                        
                        classEv = reshape(classEv,[nch*size(classEv,2), 1]);
                        eegEv = reshape(eegEv,[nch*size(eegEv,2), size(eegEv,3), size(eegEv,4)]);

                        % Algorithm
                        accuracy(k) = BCIalgorithFromCSP_2task(classTr,eegTr,classEv,eegEv,ch); 
                    end
                    ACC(s, iteration, x_cl) = mean(accuracy);
                    STD(s, iteration, x_cl) = std(accuracy);
                end
            end
        end
    end
    MATRIX_accuracy = zeros(chmax,n_subjects+3,n_class_combination);
    ACCURACY_SORT = zeros(n_subjects+3,chmax,n_class_combination);
    for i = 1:n_class_combination
        MATRIX_accuracy(:,1,i) = 1:chmax;
        MATRIX_accuracy(:,2:n_subjects+1,i) = ACC(:,:,i)';    
        MATRIX_accuracy(:,n_subjects+2,i) = mean(MATRIX_accuracy(:,2:n_subjects+1,i),2);
        MATRIX_accuracy(:,n_subjects+3,i) = std(MATRIX_accuracy(:,2:n_subjects+1,i),0,2);
        Mat_acc = sortrows(MATRIX_accuracy(:,:,i),n_subjects+3,{'descend'});
        ACCURACY_SORT(:,:,i) = Mat_acc';
    end
    
    for i = 1:n_class_combination
        channel_add(ch_selected,i) = decision_function(ACCURACY_SORT(:,:,i),n_subjects);
    end
    results(:,:,ch_selected,:) = ACCURACY_SORT;
    
    data = {struct('accuracy',ACCURACY_SORT,'channel',channel_add)};
    save(strcat('results_channels_2tasks_',num2str(ch_selected),'.mat'),'data')
    
    clearvars -except channel_add results ch_selected n_subjects chmax k_folds trials runs subjects subjects_data class_1 class_2 n_class_combination;
    clc
end

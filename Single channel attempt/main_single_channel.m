% Attempt of classification of motor imagery with single channel
%
% The inspiring paper was: 
%   "S. Ge, R. Wang, and D. Yu, “Classification of four-class motor imagery 
%    employing single-channel electroencephalography,” PloS one, vol. 9,
%    no. 6, p. e98019, 2014"
%
% The main parts are
%  - Short Time Fourier Transform (STFT) for extracting multiple bands
%  information from the single channel;
%  - Common Spatial Pattern (CSP) treating the frequency bands like spatial
%  channels and projecting them to a new space prior to extract features;
%  - Support Vector Machine (SVM) for classification.
%
% Different classes combinations are attempted.
%
% Results are in contrast with the reference study and they seem to
% indicate that successful classification of motor imagery is NOT possible
% with a single EEG channel
%
% For further details see "Metrological performance of a single-channel
% Brain-Computer Interface based on Motor Imagery" 
% (DOI: 10.1109/I2MTC.2019.8827168)
%
% IF RESULTS ARE ALREADY SAVED, JUMP TO THE END TO PLOT!
% IF YOU WANT TO EXECUTE THE ANALYSES, YOU NEED DATA FROM BCI COMPETITIONS
%  - BCI competition IV  dataset 2a: https://www.bbci.de/competition/iv/#dataset2a
%  - BCI competition III dataset 3a: https://www.bbci.de/competition/iii/#data_set_iiia
%
%
%  author:          A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2023/01/03
%

close all;
clear;
clc

% Initialize classes pairs
class_1 = [1, 1, 1, 2, 2, 3];
class_2 = [2, 3, 4, 3, 4, 4];
n_class_combination = length(class_1);

% Number of folds for Repeated Stratified Group K-Fold CV
k_folds = 6;
n_repts = 1;

% CSP components
mCSP = 2;

%% TWO TASKS dataset 2a (fixed mCSP)
% Number of subject taken into account for the competition
n_subjects = 9;
nch = 22;

% Accuracy and Standard Deviation matrix
ACC2a = zeros(n_subjects, nch, n_class_combination);
STD2a = zeros(n_subjects, nch, n_class_combination);

for s = 1:n_subjects
    % Loading dataset 2a BCI competition IV
    subj = strcat('A0',num2str(s),'T');
    if (s == 4)
        runs = 2:7;
    else
        runs = 4:9;
    end
    
    [signals, classes, fs] = extraction(subj, runs, 1:nch, 1:48, 3, 6);
    
    for x_cl = 1:n_class_combination
        % select classes
        imagery = signals(classes == class_1(x_cl) | classes == class_2(x_cl),:);
        class   = classes(classes == class_1(x_cl) | classes == class_2(x_cl));
                
        % reshape channels in third dimension
        eeg = permute(reshape(imagery, [nch size(imagery,1)/nch size(imagery,2)]),[2 3 1]);
        cla = permute(reshape(class, [nch size(class,1)/nch]),[2 1]);

        for c = 1:nch
            % single channel analysis
            eeg_sc = eeg(:,:,c);
            cla_sc = cla(:,c);

            % spectrogram
            X = STFT(eeg_sc,size(eeg_sc,1),size(eeg_sc,2));
            ch = 1:size(X,2);
            nh = length(ch);
            SPECT = permute(X,[2 1 3]);
            
            % CROSS VALIDATION (stratified k-fold partition)
            cst = cvpartition(cla_sc,'KFold', k_folds);
            cvPart.numTestSet = k_folds*n_repts;

            c_temp = cst;
            for rep = 1:n_repts
                for ind = 1:k_folds
                    cvPart.testInd{(rep-1)*k_folds+ind} = test(c_temp,ind);
                    cvPart.trainInd{(rep-1)*k_folds+ind} = training(c_temp,ind);
                end
                c_temp = repartition(c_temp);
            end
            
            accuracy = zeros(1,cvPart.numTestSet);
            for k = 1:cvPart.numTestSet
                % split data for train and test
                classEv = cla_sc(cvPart.testInd{k})*ones(1,nh);
                classTr = cla_sc(cvPart.trainInd{k})*ones(1,nh);

                eegEv = SPECT(:,cvPart.testInd{k},:);
                eegTr = SPECT(:,cvPart.trainInd{k},:);

                % reshape
                classEv = reshape(classEv,[size(classEv,1)*size(classEv,2) 1]);
                classTr = reshape(classTr,[size(classTr,1)*size(classTr,2) 1]);
                
                eegEv = reshape(eegEv,[size(eegEv,1)*size(eegEv,2) size(eegEv,3)]);
                eegTr = reshape(eegTr,[size(eegTr,1)*size(eegTr,2) size(eegTr,3)]);
                
                % CSP+SVM ALGORITHM
                % training
                [W1, W2] = CSPtrain(eegTr, classTr, ch, mCSP);
                [V, ~, ~, ~, Y] = CSPapply(eegTr, classTr, W1, W2,[],[]);

                mdl = fitcsvm(V,Y);

                % evaluation
                [Vev, ~, ~, ~, Y] = CSPapply(eegEv, classEv, W1, W2,[],[]);
                label_eval = predict(mdl,Vev);

                accuracy(k) = mean(label_eval == Y);
            end

            % results
            ACC2a(s, c, x_cl) = mean(accuracy);
            STD2a(s, c, x_cl) = std(accuracy);
        end
    end
end

%% TWO TASKS dataset 3a (fixed mCSP)
% Number of subject taken into account for the competition
subjects = ['k3b';'k6b';'l1b'];
n_subjects = 3;

% Number of folds for Repeated Stratified Group K-Fold CV
k_folds = 6;
n_repts = 1;
nch = 60;

runs = 1:6;     % 6 runs for k6b e l1b, 9 runs per k3b.  
trials = 1:20;  % 40 trial - 20 di training e 20 di test.  

% Time window of the signal
tmin = 4;
tmax = 7;

% Accuracy and Standard Deviation matrix
ACC3a = zeros(n_subjects, nch, n_class_combination);
STD3a = zeros(n_subjects, nch, n_class_combination);

for s = 1:n_subjects
    % Loading dataset 3a BCI competition III
    subj = subjects(s,:);
    pathTL = strcat(subj,'_truelabels.txt'); 
    data = divideTrainingTest(strcat(subj,'.mat'),pathTL);
    
    [signals, classes, f_samp]  = extractionTraining(data, runs, 1:nch, trials, tmin, tmax);
    
    for x_cl = 1:n_class_combination
        % select classes
        imagery = signals(classes == class_1(x_cl) | classes == class_2(x_cl),:);
        class   = classes(classes == class_1(x_cl) | classes == class_2(x_cl));
                
        % reshape channels in third dimension
        eeg = permute(reshape(imagery, [nch size(imagery,1)/nch size(imagery,2)]),[2 3 1]);
        cla = permute(reshape(class, [nch size(class,1)/nch]),[2 1]);

        for c = 1:nch
            % single channel analysis
            eeg_sc = eeg(:,:,c);
            cla_sc = cla(:,c);

            % spectrogram
            X = STFT(eeg_sc,size(eeg_sc,1),size(eeg_sc,2));
            ch = 1:size(X,2);
            nh = length(ch);
            SPECT = permute(X,[2 1 3]);
            
            % CROSS VALIDATION (stratified k-fold partition)
            cst = cvpartition(cla_sc,'KFold', k_folds);
            cvPart.numTestSet = k_folds*n_repts;

            c_temp = cst;
            for rep = 1:n_repts
                for ind = 1:k_folds
                    cvPart.testInd{(rep-1)*k_folds+ind} = test(c_temp,ind);
                    cvPart.trainInd{(rep-1)*k_folds+ind} = training(c_temp,ind);
                end
                c_temp = repartition(c_temp);
            end
            
            accuracy = zeros(1,cvPart.numTestSet);
            for k = 1:cvPart.numTestSet
                % split data for train and test
                classEv = cla_sc(cvPart.testInd{k})*ones(1,nh);
                classTr = cla_sc(cvPart.trainInd{k})*ones(1,nh);

                eegEv = SPECT(:,cvPart.testInd{k},:);
                eegTr = SPECT(:,cvPart.trainInd{k},:);

                % reshape
                classEv = reshape(classEv,[size(classEv,1)*size(classEv,2) 1]);
                classTr = reshape(classTr,[size(classTr,1)*size(classTr,2) 1]);
                
                eegEv = reshape(eegEv,[size(eegEv,1)*size(eegEv,2) size(eegEv,3)]);
                eegTr = reshape(eegTr,[size(eegTr,1)*size(eegTr,2) size(eegTr,3)]);
                
                try
                    % CSP+SVM ALGORITHM
                    % training
                    [W1, W2] = CSPtrain(eegTr, classTr, ch, mCSP);
                    [V, ~, ~, ~, Y] = CSPapply(eegTr, classTr, W1, W2,[],[]);

                    mdl = fitcsvm(V,Y);

                    % evaluation
                    [Vev, ~, ~, ~, Y] = CSPapply(eegEv, classEv, W1, W2,[],[]);
                    label_eval = predict(mdl,Vev);

                    accuracy(k) = mean(label_eval == Y);
                catch
                    accuracy(k) = NaN;
                end
            end

            % results
            ACC3a(s, c, x_cl) = nanmean(accuracy);
            STD3a(s, c, x_cl) = nanstd(accuracy);
        end
    end
end

%% saving
% save('Saved_Results.mat','ACC2a','STD2a','ACC3a','STD3a');

%% plotting
% plotting 2a
fig2a = figure;
for s = 1:9
    subplot(3,3,s)
    for x_cl = 1:n_class_combination
        hold on
        plot(100*ACC2a(s,:,x_cl))
    end
    plot([1 22],[50 50],'r--')
    xlim([1 22])
    ylim([0 100])
    title(strcat('A0',num2str(s)))
    grid
    ytickformat('%.0f')
    ylabel('accuracy / %')
    xlabel('channel')
end
% legend('LvsR','LvsF','LvsT','RvsF','RvsT','FvsT','Location','SouthWest')

% plotting 3a
fig3a = figure;
chs = [2 3 4 28 31 34];       % nanchino version (Fp1, Fpz, Fp2, C3, Cz, C4)
nchs = length(chs);
for s = 1:3
    subplot(1,3,s)
    for x_cl = 1:n_class_combination
        hold on
        plot(ACC3a(s,chs,x_cl))
    end
    plot([1 nchs],[0.5 0.5],'r--')
    xlim([1 nchs])
    ylim([0 1])
    legend('LR','LF','LT','RF','RT','FT','Location','SouthWest')
end


%% TWO TASKS dataset 3a (variable mCSP)
% Number of subject taken into account for the competition
subjects = ['k3b';'k6b';'l1b'];
n_subjects = 3;

% Number of folds for Repeated Stratified Group K-Fold CV
k_folds = 6;
n_repts = 1;
chs = [2 3 4 28 31 34];
nch = length(chs);

runs = 1:6;     % 6 runs for k6b e l1b, 9 runs per k3b.  
trials = 1:20;  % 40 trial - 20 di training e 20 di test.  

% Time window of the signal
tmin = 4;
tmax = 7;

% variable mCSP
mCSP = 1:10;
nCSP = length(mCSP);

% Accuracy and Standard Deviation matrix
ACC3a = zeros(n_subjects, nch, n_class_combination, nCSP);
STD3a = zeros(n_subjects, nch, n_class_combination, nCSP);

for s = 1:n_subjects
    % Loading dataset 3a BCI competition III
    subj = subjects(s,:);
    pathTL = strcat(subj,'_truelabels.txt'); 
    data = divideTrainingTest(strcat(subj,'.mat'),pathTL);
    
    [signals, classes, f_samp]  = extractionTraining(data, runs, chs, trials, tmin, tmax);
    
    for x_cl = 1:n_class_combination
        % select classes
        imagery = signals(classes == class_1(x_cl) | classes == class_2(x_cl),:);
        class = classes(classes == class_1(x_cl) | classes == class_2(x_cl));
                
        % reshape channels in third dimension
        eeg = permute(reshape(imagery, [nch size(imagery,1)/nch size(imagery,2)]),[2 3 1]);
        cla = permute(reshape(class, [nch size(class,1)/nch]),[2 1]);

        for c = 1:nch
            % single channel analysis
            eeg_sc = eeg(:,:,c);
            cla_sc = cla(:,c);

            % spectrogram
            X = STFT(eeg_sc,size(eeg_sc,1),size(eeg_sc,2));
            ch = 1:size(X,2);
            nh = length(ch);
            SPECT = permute(X,[2 1 3]);
            
            % CROSS VALIDATION (stratified k-fold partition)
            cst = cvpartition(cla_sc,'KFold', k_folds);
            cvPart.numTestSet = k_folds*n_repts;

            c_temp = cst;
            for rep = 1:n_repts
                for ind = 1:k_folds
                    cvPart.testInd{(rep-1)*k_folds+ind} = test(c_temp,ind);
                    cvPart.trainInd{(rep-1)*k_folds+ind} = training(c_temp,ind);
                end
                c_temp = repartition(c_temp);
            end
            
            accuracy = zeros(nCSP,cvPart.numTestSet);
            for k = 1:cvPart.numTestSet
                % split data for train and test
                classEv = cla_sc(cvPart.testInd{k})*ones(1,nh);
                classTr = cla_sc(cvPart.trainInd{k})*ones(1,nh);

                eegEv = SPECT(:,cvPart.testInd{k},:);
                eegTr = SPECT(:,cvPart.trainInd{k},:);

                % reshape
                classEv = reshape(classEv,[size(classEv,1)*size(classEv,2) 1]);
                classTr = reshape(classTr,[size(classTr,1)*size(classTr,2) 1]);
                
                eegEv = reshape(eegEv,[size(eegEv,1)*size(eegEv,2) size(eegEv,3)]);
                eegTr = reshape(eegTr,[size(eegTr,1)*size(eegTr,2) size(eegTr,3)]);
                
                for csp = 1:nCSP
                    try
                        % CSP+SVM ALGORITHM
                        % training
                        [W1, W2] = CSPtrain(eegTr, classTr, ch, mCSP(csp));
                        [V, ~, ~, ~, Y] = CSPapply(eegTr, classTr, W1, W2,[],[]);

                        mdl = fitcsvm(V,Y);

                        % evaluation
                        [Vev, ~, ~, ~, Y] = CSPapply(eegEv, classEv, W1, W2,[],[]);
                        label_eval = predict(mdl,Vev);

                        accuracy(csp,k) = mean(label_eval == Y);
                    catch
                        accuracy(csp,k) = NaN;
                    end
                end
            end
            
            % results
            ACC3a(s, c, x_cl, :) = nanmean(accuracy,2);
            STD3a(s, c, x_cl, :) = nanstd(accuracy,[],2);
        end
    end
end

% plotting 3a
figure
nchs = length(chs);
for s = 1:n_subjects
    for p = 1:n_class_combination
        subplot(n_subjects,n_class_combination,(s-1)*n_class_combination + p)
        for ch = 1:nchs
            hold on
            plot(permute(ACC3a(s,ch,p,:),[4 3 2 1]))
        end
        plot([1 nCSP],[0.5 0.5],'r--')
        xlim([1 nCSP])
        ylim([0 1])
    end
end
legend('Fp1', 'Fpz', 'Fp2', 'C3', 'Cz', 'C4','Location','SouthWest')

%% FOUR TASKS dataset 3a (variable mCSP)
% Number of subject taken into account for the competition
subjects = ['k3b';'k6b';'l1b'];
n_subjects = 3;

% Number of folds for Repeated Stratified Group K-Fold CV
k_folds = 6;
n_repts = 1;
chs = [2 3 4 28 31 34];
nch = length(chs);

runs = 1:6;     % 6 runs for k6b e l1b, 9 runs per k3b.  
trials = 1:20;  % 40 trial - 20 di training e 20 di test.  

% Time window of the signal
tmin = 4;
tmax = 7;

% variable mCSP
mCSP = 1:10;
nCSP = length(mCSP);

% Accuracy and Standard Deviation matrix
ACC3a = zeros(n_subjects, nch, nCSP);
STD3a = zeros(n_subjects, nch, nCSP);

for s = 1:n_subjects
    % Loading dataset 3a BCI competition III
    subj = subjects(s,:);
    pathTL = strcat(subj,'_truelabels.txt'); 
    data = divideTrainingTest(strcat(subj,'.mat'),pathTL);
    
    [signals, classes, f_samp]  = extractionTraining(data, runs, chs, trials, tmin, tmax);
    
    % four classes
    imagery = signals;
    class = classes;

    % reshape channels in third dimension
    eeg = permute(reshape(imagery, [nch size(imagery,1)/nch size(imagery,2)]),[2 3 1]);
    cla = permute(reshape(class, [nch size(class,1)/nch]),[2 1]);

    for c = 1:nch
        % single channel analysis
        eeg_sc = eeg(:,:,c);
        cla_sc = cla(:,c);

        % spectrogram
        X = STFT(eeg_sc,size(eeg_sc,1),size(eeg_sc,2));
        ch = 1:size(X,2);
        nh = length(ch);
        SPECT = permute(X,[2 1 3]);

        % CROSS VALIDATION (stratified k-fold partition)
        cst = cvpartition(cla_sc,'KFold', k_folds);
        cvPart.numTestSet = k_folds*n_repts;

        c_temp = cst;
        for rep = 1:n_repts
            for ind = 1:k_folds
                cvPart.testInd{(rep-1)*k_folds+ind} = test(c_temp,ind);
                cvPart.trainInd{(rep-1)*k_folds+ind} = training(c_temp,ind);
            end
            c_temp = repartition(c_temp);
        end

        accuracy = zeros(nCSP,cvPart.numTestSet);
        for k = 1:cvPart.numTestSet
            % split data for train and test
            classEv = cla_sc(cvPart.testInd{k})*ones(1,nh);
            classTr = cla_sc(cvPart.trainInd{k})*ones(1,nh);

            eegEv = SPECT(:,cvPart.testInd{k},:);
            eegTr = SPECT(:,cvPart.trainInd{k},:);

            % reshape
            classEv = reshape(classEv,[size(classEv,1)*size(classEv,2) 1]);
            classTr = reshape(classTr,[size(classTr,1)*size(classTr,2) 1]);

            eegEv = reshape(eegEv,[size(eegEv,1)*size(eegEv,2) size(eegEv,3)]);
            eegTr = reshape(eegTr,[size(eegTr,1)*size(eegTr,2) size(eegTr,3)]);

            for csp = 1:nCSP
                try
                    % CSP+SVM ALGORITHM
                    % training
                    [W1, W2, W3, W4] = CSPtrain(eegTr, classTr, ch, mCSP(csp));
                    [V1, V2, V3, V4, Y] = CSPapply(eegTr, classTr, W1, W2, W3, W4);

                    %mdl = fitcecoc([V1 V2 V3 V4],Y);

                    % evaluation
                    [Vev1, Vev2, Vev3, Vev4, Yev] = CSPapply(eegEv, classEv, W1, W2, W3, W4);
                    %label_eval = predict(mdl,[Vev1, Vev2, Vev3, Vev4]);

                    % Naive Bayesian Parzen Window classifier
                    nt = length(Yev);
                    proba = zeros(nt,1);
                    index = zeros(nt,1);
                    ind = index;
                    for i = 1:nt
                        pwx1 = NBPW(V1, Y, Vev1(i,:), 1);
                        pwx2 = NBPW(V2, Y, Vev2(i,:), 2);
                        pwx3 = NBPW(V3, Y, Vev3(i,:), 3);
                        pwx4 = NBPW(V4, Y, Vev4(i,:), 4);
                        [proba(i), index(i)] = max([pwx1 pwx2 pwx3 pwx4]);
                    end
                    ind(index==1) = 1;
                    ind(index==2) = 2;
                    ind(index==3) = 3;
                    ind(index==4) = 4;

                    % Classification accuracy
                    accuracy(csp,k) = mean(ind == Yev);
                catch
                    accuracy(csp,k) = NaN;
                end
            end
        end

        % results
        ACC3a(s, c, :) = nanmean(accuracy,2);
        STD3a(s, c,:) = nanstd(accuracy,[],2);
    end
end

% plotting 3a
figure
nchs = length(chs);
for s = 1:n_subjects
    subplot(n_subjects,1,s)
    for ch = 1:nchs
        hold on
        plot(permute(ACC3a(s,ch,:),[3 2 1]))
    end
    plot([1 nCSP],[0.5 0.5],'r--')
    xlim([1 nCSP])
    ylim([0 1])
end
legend('Fp1', 'Fpz', 'Fp2', 'C3', 'Cz', 'C4','Location','SouthWest')
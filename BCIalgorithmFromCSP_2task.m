function [accuracy, predicted] = BCIalgorithmFromCSP_2task(classTrain, eegTrain, classTest, eegTest, ch)
% BCIALGORITHMFROMCSP_2TASK encloses the piece of the FBCSP algorithm after
% the filter-bank for both training and evaluation of the model, hence it
% implements the Common Spatial Pattern (CSP) + the Features Selection
% (MIBIF) + the Classifier (NBPW, naive bayesian parzen window)
%
%   INPUT:
%   'classTrain' is the 1D array containing class labels for the training
%   data 'eegTrain';
%
%   'eegTrain' is the 3D array containing training EEG signals; its
%   dimensions must be #trials x #samples x #bands; if multiple channels
%   are available, signals associated to the same trial and different
%   channels are consecutive: hence #trials is actually #trials x #ch; the
%   labels associated to these signals will be organized accordingly. Note
%   that data must be already filtered by the filter bank;
%
%   'classTest' is the 1D array containing class labels for the test data
%   'eegTest'; they should be unknown when using the BCI, but here they are
%   used to evaluate the model performance;
%
%   'eegTest' is the 3D array containing test EEG signals; its
%   dimensions must be #trials x #samples x #bands; if multiple channels
%   are available, signals associated to the same trial and different
%   channels are consecutive: hence #trials is actually #trials x #ch; the
%   labels associated to these signals will be organized accordingly. Note
%   that data must be already filtered by the filter bank;
%
%   OUTPUT:
%   'accuracy' is the model performance indicator calculated by comparing
%   the known test data labels with the labels guessed by the trained
%   model;
%
%   'predicted' is 1D array with predicted labels for the 'eegTest' data;
%   if a single trial is furnished as test, this consists of the single
%   label associated to the signal to be classified, and the accuracy can
%   either be 0% or 100% (wrong or right).
%
%  authors:         A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2020/09/03
%
%   NOTE: future upgrades of this script could comprehend returning the
%   parameters identifying the model for classifying new data
%

    k_MIBIF = 5;
    
    % TRAINING
    class = classTrain;
    eeg = eegTrain;
    two_class = unique(class);
    class1 = two_class(1);
    class2 = two_class(2);
    
    % SPATIAL FILTERING
    if (length(ch) == 1)
        % No spatial filter in case of a single channel
        nb = size(eeg,3);
        V = zeros(size(eeg,1),nb);

        for i = 1:nb
            E = eeg(:,:,i);
            C = E*E';
            V(:,i) = log(diag(C)/trace(C));
        end

        V1 = V;
        V2 = V;
        Y = class;

        m = 0;          % with this value, the MIBIF will understand that there is no spatial filter applied
    else
        % The Common Spatial Pattern algorithm is applicable
        if (length(ch) < 4)
            % but 'm' cannot be 2 in this case
            m = 1;
        else
            m = 2;
        end
        
        [W1, W2] = CSPtrain(eeg, class, ch, m);
        [V1, V2, ~, ~, Y] = CSPapply(eeg, class, W1, W2, [], []);
    end

    % FEATURES SELECTION
    % Indexes with maximum Mutual-Information and their complementary
    I1 = MIBIF(V1,Y,class1,m,k_MIBIF);
    I2 = MIBIF(V2,Y,class2,m,k_MIBIF);

    % Selection
    f1 = V1(:,I1);
    f2 = V2(:,I2);

    % CLASSIFIER TRAINING
    % Composite vector (Naive Bayesian training)
    f = [f1, f2];
    cl = Y;

    % EVALUATION    
    class = classTest;
    eeg = eegTest;
    % SPATIAL FILTERING
    if (length(ch) == 1)
        % No spatial filter in case of a single channel
        nb = size(eeg,3);
        V = zeros(size(eeg,1),nb);
        for i = 1:nb
            E = eeg(:,:,i);
            C = E*E';
            V(:,i) = log(diag(C)/trace(C));
        end
        V1 = V;
        V2 = V;
        Y = class;
    else
        [V1, V2, ~, ~, Y] = CSPapply(eeg, class, W1, W2, [], []);
    end
    
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
    
    % Returning predicted labels for test data
    predicted = ind;
end
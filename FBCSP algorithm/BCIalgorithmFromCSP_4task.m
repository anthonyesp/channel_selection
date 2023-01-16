function accuracy = BCIalgorithmFromCSP_4task(classTrain,eegTrain,classTest,eegTest,ch)
% BCIALGORITHMFROMCSP_4TASK encloses the piece of the FBCSP algorithm after
% the filter-bank for both training and evaluation of the model, hence it
% implements the Common Spatial Pattern (CSP) + the Features Selection
% (MIBIF) + the Classifier (NBPW, naive bayesian parzen window) in their
% multi-class version (4 classes in particular)
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
        V3 = V;
        V4 = V;
        Y = class;

        m = 0;          % with this value, the MIBIF will understand that there is no spatial filter applied
    else
        if (length(ch) < 4)
            % The Common Spatial Pattern algorithm is applicable
            % but 'm' cannot be 2
            m = 1;
        else
            m = 2;
        end
        [W1, W2, W3, W4] = CSPtrain(eeg, class, ch, m);
        [V1, V2, V3, V4, Y] = CSPapply(eeg, class, W1, W2, W3, W4);
    end

    % FEATURES SELECTION
    % Indexes with maximum Mutual-Information and their complementary
    I1 = MIBIF(V1,Y,1,m,k_MIBIF);
    I2 = MIBIF(V2,Y,2,m,k_MIBIF);
    I3 = MIBIF(V3,Y,3,m,k_MIBIF);
    I4 = MIBIF(V4,Y,4,m,k_MIBIF);

    % Selection
    f1 = V1(:,I1);
    f2 = V2(:,I2);
    f3 = V3(:,I3);
    f4 = V4(:,I4);

    % CLASSIFIER TRAINING
    % Composite vector (Naive Bayesian training)
    f = [f1, f2, f3, f4];
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
        V3 = V;
        V4 = V;
        Y = class;
    else
        [V1, V2, V3, V4, Y] = CSPapply(eeg, class, W1, W2, W3, W4);
    end
    
    % FEATURES SELECTION
    % Selection
    feval1 = V1(:,I1);
    feval2 = V2(:,I2);
    feval3 = V3(:,I3);
    feval4 = V4(:,I4);

    % CLASSIFICATION
    % Composite vector
    feval = [feval1, feval2, feval3, feval4];
    cleval = Y;

    % Naive Bayesian Parzen Window classifier
    nt = length(cleval);
    proba = zeros(nt,1);
    index = zeros(nt,1);
    ind = index;
    for i = 1:nt
        pwx1 = NBPW(f1, cl, feval1(i,:), 1);
        pwx2 = NBPW(f2, cl, feval2(i,:), 2);
        pwx3 = NBPW(f3, cl, feval3(i,:), 3);
        pwx4 = NBPW(f4, cl, feval4(i,:), 4);
        [proba(i), index(i)] = max([pwx1 pwx2 pwx3 pwx4]);
    end
    ind(index==1) = 1;
    ind(index==2) = 2;
    ind(index==3) = 3;
    ind(index==4) = 4;

    % Classification accuracy
    accuracy = (sum(ind == cleval)/nt)*100;
end
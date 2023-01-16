function [W1, W2, W3, W4, C1, C2, n1, n2, C3, C4, n3, n4] = CSPtrain(eeg, class, ch, m)
% This function trains the CSP block according to input data.
% The input classes can be 2 or 4 in the current implementation.
%
%   INPUT:
%   'eeg' is an array with EEG filtered at different pass-bands;
%   'class' is a column vector with the class of each EEG signal;
%   'ch' are the channels to consider for the eeg signal extraction;
%   'm' is the number of CSP components to consider.
%
%   OUTPUT:
%   'Wi' is the CSP projection matrix related to class i, with i in range (1,4);
%   'Ci' is the covariance matrix related to class i, with i in range (1,4);
%   'ni' is the trials number to calculate the covariance matrix related to class i, with i in range (1,4);
%
%
%  authors:         A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2020/11/30

    % data reorganization
    [X1,X2,X3,X4] = separation(eeg,class,ch);
    
    % classes detection
    flag = [~isempty(X1), ~isempty(X2), ~isempty(X3), ~isempty(X4)];
    
    nb = size(X1,4);
    nch = size(X1,1);
    if (sum(flag) == 2)
        % TWO-CLASSES VERSION
        % take the two 'X'
        if (flag(1))
            Xa = X1;
            if (flag(2))
                Xb = X2;
            elseif (flag(3))
                Xb = X3;
            else
                Xb = X4;
            end
        elseif (flag(2))
            Xa = X2;
            if (flag(3))
                Xb = X3;
            else
                Xb = X4;
            end
        else
            Xa = X3;
            Xb = X4;
        end
        
        % check 'm' versus available channels
        if (nch < 2)
            W1 = 1;
            W2 = 1;
        else
            if (nch < 4)
                m = 1;
            end
            
            % projection matrices
            W1 = zeros(nch,2*m,nb);
            W2 = zeros(nch,2*m,nb);
            C1 = zeros(nch,nch,nb);
            C2 = zeros(nch,nch,nb);
            % CSP matrices
            for i = 1:nb
                [W1(:,:,i), W2(:,:,i), C1(:,:,i), C2(:,:,i), n1, n2] = CSP_2task(Xa(:,:,:,i),Xb(:,:,:,i), m);
            end
        end
        
        W3 = [];
        W4 = [];
    
    elseif (sum(flag) == 4)
        % check 'm' versus available channels
        if (nch < 2)
            W1 = 1;
            W2 = 1;
            W3 = 1;
            W4 = 1;
        else
            if (nch < 4)
                m = 1;
            end
            
            % FOUR-CLASSES VERSION
            % projection matrices
            W1 = zeros(nch,2*m,nb);
            W2 = zeros(nch,2*m,nb);
            W3 = zeros(nch,2*m,nb);
            W4 = zeros(nch,2*m,nb);
            C1 = zeros(nch,nch,nb);
            C2 = zeros(nch,nch,nb);
            C3 = zeros(nch,nch,nb);
            C4 = zeros(nch,nch,nb);

            for i = 1:nb
                % CSP matrices
                [W1(:,:,i), W2(:,:,i), W3(:,:,i), W4(:,:,i), C1(:,:,i), C2(:,:,i), C3(:,:,i), C4(:,:,i), n1, n2, n3, n4] = CSP_4task(X1(:,:,:,i),X2(:,:,:,i),X3(:,:,:,i),X4(:,:,:,i), m);
            end
        end
    else
        error('Current CSP implementation only allows 2 or 4 classes!');
    end    
end
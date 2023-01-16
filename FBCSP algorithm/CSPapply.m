function [V1, V2, V3, V4, Y] = CSPapply(eeg, class, W1, W2, W3, W4)
% This function performs spatial filtering using the CSP algorithm.

%   INPUT:
%   'step' can only be equal to 'training' or 'evaluation';
%   'eeg' is an array with eeg filtered at different pass-bands;
%   'class' is a coloumn vector with the class of each eeg signal
%   contained in 'imagery'; hence, its number of rows is equal to the
%   number of rows of 'imagery';
%   'ch' are the channels to consider for the eeg signal extraction;
%   'W1' is the projection matrix associated to the class '1';
%   'W2' is the projection matrix associated to the class '2';
%   'W3' is the projection matrix associated to the class '3';
%   'W4' is the projection matrix associated to the class '4'.
%
%   OUTPUT:
%   'V1' are the features extracted from 'eeg' through W1
%   'V2' are the features extracted from 'eeg' through W2
%   'V3' are the features extracted from 'eeg' through W3
%   'V4' are the features extracted from 'eeg' through W4
%   'Y' is the new labels array separated per trial
%
%
%  authors:         A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2020/05/28

    m = size(W1,2)/2;
    nb = size(W1,3);
    nch = size(W1,1);
    nt = size(eeg,1)/nch;
    
    V1 = zeros(nt,nb*2*m);
    V2 = zeros(nt,nb*2*m);
    V3 = zeros(nt,nb*2*m);
    V4 = zeros(nt,nb*2*m);
    
    eeg = permute(reshape(eeg,[nch, nt, size(eeg,2), size(eeg,3)]),[1,3,2,4]);
    for i = 1:nb
        % Features
        for j = 1:nt
            
            E = eeg(:,:,j,i);

            % first task
            if (isempty(W1))
                V1 = [];
            else
                C1 = W1(:,:,i)'*E*E'*W1(:,:,i);
                V1(j,(i-1)*2*m+1:i*2*m) = transpose(log(diag(C1)/trace(C1)));
            end
            
            % second task
            if (isempty(W2))
                V2 = [];
            else
                C2 = W2(:,:,i)'*E*E'*W2(:,:,i);
                V2(j,(i-1)*2*m+1:i*2*m) = transpose(log(diag(C2)/trace(C2)));
            end
            
            % third task
            if (isempty(W3))
                V3 = [];
            else
                C3 = W3(:,:,i)'*E*E'*W3(:,:,i);
                V3(j,(i-1)*2*m+1:i*2*m) = transpose(log(diag(C3)/trace(C3)));
            end
            
            % forth task
            if (isempty(W4))
                V4 = [];
            else
                C4 = W4(:,:,i)'*E*E'*W4(:,:,i);
                V4(j,(i-1)*2*m+1:i*2*m) = transpose(log(diag(C4)/trace(C4)));
            end
        end
    end

    % new label array (if needed)
    if(class ~= 0)
        Y = reshape(class,[nch nt]);
        Y = Y(2,:)';
    else
        Y = 0;
    end
end


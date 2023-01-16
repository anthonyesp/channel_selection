function [XL, XR, XF, XT] = separation(eeg,class,ch)
% SEPARATION separates the signals per class and reshapes them in a 4D
% array with channels-time-trials-band
%
%   INPUT:
%   'eeg' is a 3D array with EEG filtered at different pass-bands;
%   'class' is a column vector with the class of each EEG signal;
%   'ch' are the channels to consider for the eeg signal extraction;
%
%   OUTPUT:
%   'XL' is the 4D array with filtered EEG related to class 'left';
%   'XR' is the 4D array with filtered EEG related to class 'right';
%   'XF' is the 4D array with filtered EEG related to class 'feet';
%   'XT' is the 4D array with filtered EEG related to class 'tongue';
%
%
%  authors:         A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2020/05/12

    % Separation per class
    nch = length(ch);
    eeg = reshape(eeg,[nch size(eeg,1)/nch size(eeg,2) size(eeg,3)]);
    class = reshape(class,[nch size(class,1)/nch]);

    per = [1 3 2 4];
    XL =  permute(eeg(:,class(1,:) == 1,:,:),per);
    XR =  permute(eeg(:,class(1,:) == 2,:,:),per);
    XF =  permute(eeg(:,class(1,:) == 3,:,:),per);
    XT =  permute(eeg(:,class(1,:) == 4,:,:),per);
end
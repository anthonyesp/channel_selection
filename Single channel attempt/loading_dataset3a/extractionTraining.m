function [signals, class, f_samp] = extractionTraining(data, run, ch, trial, tmin, tmax)
% EXTRACTION extracts the desired EEG signal(s) during motor imagery.
%
%   INPUT:
%   'path' is the .mat dataset name from which EEG signals are to extract;
%   'run' are the runs to consider;
%   'ch' are the channels to consider for the EEG signal;
%   'trial' are the trials to consider for each specified run;
%   'tmin' is the starting point of the time window to consider, furnished
%   as an offset from the trial start [seconds]
%   'tmax' is the ending point of the time window to consider, furnished
%   as an offset from the trial start [seconds]
%
%   OUTPUT:
%   'signals' is an array containing EEG signals in the time window;
%   the array rows corrispond to trails: they are grouped by channel 
%   and they can be divided by runs; the coloumns are time samples.
%   
%   'class' is a coloumn vector with the class of each EEG signal
%   contained in 'signals'; hence, its number of rows is equal to the
%   number of rows of 'signals.
%
%   'f_sampling' is the sampling frequenciy associated to the signals.
%
%   NOTE:
%   for a better understanding, refer to the BCI Competition IV description
%   of "dataset 2a" containing 4-class motor imagery eeg signals; also see
%   the following checks on the input to understand their allowed ranges.
%
%
%  authors:         S. Montella, A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2020/08/25


% Input checks
    % 'run' must be a 1-D array
    if (length(find(size(run) ~= 1)) > 1)
        error('"run" must be a 1-D array.')
    end

    % 'trial' must be a 1-D array
    if (length(find(size(trial) ~= 1)) > 1)
        error('"trial" must be a 1-D array.')
    end

    % 'ch' must be a 1-D array
    if (length(find(size(ch) ~= 1)) > 1)
        error('"ch" must be a 1-D array.')
    end



% Initializations
    % Sampling frequency (supposed equal for every trial in 'path')
    f_samp = data.HDR.SampleRate; 
    
    % Time window to consider for the analysis
    n_samp = (tmax - tmin)*f_samp;    % number of samples of time window
    
% Extraction of desired signals
    % 'run', 'trial' and 'ch' lengths
    m = length(run);
    n = length(trial);
    k = length(ch);

    % Extracting signal
    class = zeros(m*n*k,1);
    signals = zeros(m*n*k,n_samp);
    
    for i = 1:m*n
        tri = data.TrialTrain(i) ; 
        t_start = tri + tmin*f_samp ;  
        t_stop  = tri + tmax*f_samp ; 
       
        for h = 1:k
            index = (i-1)*k + h ; 
            class(index) = data.ClassTrain(i); 
            signals(index,:) = data.s(t_start:t_stop-1, ch(h));
        end
    end
end


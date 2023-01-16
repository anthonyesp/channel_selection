function [filtered_eeg, HD] = filterBank(eeg, fsamp, varargin)
% FILTERBANK decomposes the EEG into multiple frequency pass bands using 
% causal Chebyshev Type II filter.
%
%   INPUT
%   'eeg' is a single electroencephalographic signal in time domain;
%   'fsamp' is the sampling frequency of the digital eeg;
%   'varargin' eventually contains the filter saved from previous 
%    iterations to speed up the processing
%
%   OUTPUT
%   'filtered_eeg' is an array with eeg filtered at different pass-bands;
%   'HD' is the filter save for next iterations to speed up
%
%
%  authors:         A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2020/11/30
%

% check of the variable input argument
if (nargin > 3)
    error('Too many input arguments!');
elseif (nargin == 3)
    HD = varargin{1};
    
    % something is passed, but it could be empty
    if(isempty(HD))
        create_filter = 1;
        clear HD;
    else
        create_filter = 0;
    end
elseif (nargin < 2)
    error('Not enough input arguments!');
end

    % filter parameters
    order = 10;                 % * chosen arbitrarily
    attenuation = 50;           % * chosen arbitrarily
    
    % filter bands
    fcutL = 4;                        % low  cut off frequency
    fcutH = 40;                       % high cut off frequency
    fshif = 2;                        % shift for bands overlap
    fband = 4;                        % band width
    
    % filtered signal init
    n_bands = floor(((fcutH-fcutL) - fband)/fshif + 1);
    filtered_eeg = zeros([size(eeg) n_bands]);
    
    % check for not-a-numbers in the data and replace it with zeros
    signal_nan = find(isnan(eeg));
    if (~isempty(signal_nan))
        nnan = length(signal_nan);
        warning(strcat(num2str(nnan),' NaN were found in the signal to filter! They were replaced with 0.')); 
        eeg(signal_nan) = 0;
    end
    
    % filtering
    i = 1;
    while (i <= n_bands)
        % i-th pass-band
        f_low = fcutL + fshif*(i-1);
        f_high = f_low + fband;

        if (nargin == 2 || create_filter)            
            % Chebyshev type II filter
            [z,p,k] = cheby2(order,attenuation,2*[f_low f_high]/fsamp);
            [sos,g] = zp2sos(z,p,k);
            HD(i) = dfilt.df2sos(sos,g);
        end
        
        % filtering each eeg from different channels/trials/runs
        for j = 1:size(eeg,1)
            filtered_eeg(j,:,i) = filtfilt(HD(i).sosMatrix,HD(i).ScaleValues,eeg(j,:));
        end
        
        i = i + 1;
    end
end


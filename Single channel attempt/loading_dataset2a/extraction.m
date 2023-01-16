function [signals, class, f_samp, run, ch, trials] = extraction(path, run, ch, trials, tmin, tmax, varargin)
% EXTRACTION extracts the desired EEG signal(s) from a data file save in
% Graz format (in particular BCI competition IV dataset 2a)
%
%   INPUT:
%    - 'path' is the .mat file from which EEG signals must be extracted;
%    - 'run' contains the runs to consider, leave empty for all;
%    - 'ch' contains the channels to consider, leave empty for all;
%    - 'trials' are the trials to consider per run, leave empty for all;
%    - 'tmin' is the starting point of the time window to consider,
%    consisting of an offset from the trial start [in seconds];
%    - 'tmax' is the ending point of the time window to consider,
%    consisting of an offset from the trial start [in seconds];
%
%   VARARGIN:
%	'artif' determines how to treat artifacts: 
%       1 - discard marked artifacts
%		2 - apply an artifact removal technique (currently only ASR)
%       3 - discard marked artifacts and remove others
%
%   OUTPUT:
%   'signals' is an array containing EEG signals in the time window;
%   the array rows corrispond to trails: they are grouped by channel 
%   and they can be divided by runs; the coloumns are time samples.
%   
%   'class' is a column vector with the class of each EEG signal
%   contained in 'signals'; hence, its number of rows is equal to the
%   number of rows of 'signals.
%
%   'f_sampling' is the sampling frequenciy associated to the signals.
%
%   'run', 'ch' and 'trials' are useful outputs only when they are left
%   empty at the input. Note that 'run' and 'ch' are 1-D arrays, but 
%   'trials' can be a 2-D array when a different number of trials is
%   available per each run.
%
%   NOTE:
%   for a better understanding, refer to the BCI Competition IV description
%   of "dataset 2a" containing 4-class motor imagery eeg signals; also see
%   the following checks on the input to understand their allowed ranges.
%
%
%  authors:         A. Esposito, A. Natalizio
%  correspondence:  anthony.esp@live.it
%  last update:     2021/07/12

% Varargin check
if (nargin > 7)
    error('Too many input arguments!');
elseif (nargin == 7)
    artif = varargin{1};
    
    % saturation to avoid unused numbers for this variable
    if (artif > 3)
        artif = 3;
    elseif (artif < 1)
        artif = 0;
    end
else
    artif = 0;
end

% Input checks and dimensions derivation
    % loading 'path' and checking if empty
    data = load(path);
    if (isempty(data))
        error('Path ''%s'' contains no data.\nVerify that "path" name is correct.',path);
    end 
    
    % check is the runs must be automatically determined
    if (isempty(run))
        run = 1:length(data.data);
    end
    
    % 'run' must be a 1-D array
    if (length(find(size(run) ~= 1)) > 1)
        error('"run" must be a 1-D array.')
    end
    
    m = length(run);

    
    % check is the channels must be automatically determined
    if (isempty(ch))
        % assuming the same channels for all runs
        ch = 1:size(data.data{1,run(1)}.X,2);
    end

    % 'ch' must be a 1-D array
    if (length(find(size(ch) ~= 1)) > 1)
        error('"ch" must be a 1-D array.')
    end  

    k = length(ch);

    
    % check is the trials must be automatically determined
    if (isempty(trials))
        % first find the number of trials per each run
        nt = zeros(m,1);
        for j = 1:m
            nt(j) = length(data.data{1,run(j)}.trial);
        end
        
        % then fill all the trials leaving nans in empty parts
        trials = NaN(m,max(nt));
        for j = 1:m
            trials(j,1:nt) = 1:nt;
        end
    end
    
    % 'trial' must be at most a 2-D array
    if (length(find(size(trials) ~= 1)) > 2)
        error('"trials" must be at most a 2-D array.')
    elseif(length(find(size(trials) ~= 1)) == 2)
        % a 2-D array is used when different number of trials are available
        % per each run
        if (size(trials,1) ~= m)
            error('number of rows for "trials" must equal the number of runs when it is a 2-D array')
        end
    elseif(length(find(size(trials) ~= 1)) == 1)
        % if instead a 1-D array is provided, replicate the same trials for
        % all runs to make the code work properly!
        trials = ones(m,1).*trials;
    end 
    
    n = sum(~isnan(trials),2);
    
% Initializations
    % Sampling frequency (supposed equal for every trial in 'path')
    f_samp = data.data{1,run(1)}.fs;
    
    % Time window to consider for the analysis
    n_samp = floor((tmax - tmin)*f_samp);   % number of samples of time window
    
% Extraction of desired signals
    class = zeros(sum(n)*k,1);
    artifacts = zeros(sum(n)*k,1);
    signals = zeros(sum(n)*k,n_samp);

    for j = 1:m
        % artifact removal with Artifact Subspace Reconstruction
        if (artif == 2 || artif == 3)
            try
                X_artif = data.data{1, run(j)}.X';
                X_artif = X_artif(:,data.data{1, run(j)}.trial(1):end);
                
                % file with channel locations
                if (k == 8)
                    chanlocs = 'helmate.ced';
                    warning('Helmate channel location file used for 8 channels');
                elseif (k == 22)
                    chanlocs = 'BCIcompetitionIV2a.ced';
                    warning('BCI competition IV 2a channel location file used for 22 channels');
                else
                    error(['please provide a channel location file to use ASR with ', num2str(k), ' channels']);
                end

                EEG_artif = pop_editset(eeg_emptyset,'setname','eeg','data',X_artif,...
                'chanlocs',chanlocs,'dataformat','array','srate',f_samp,'nbchan',k);

                ASR = clean_artifacts(EEG_artif,'WindowCriterion','off','chancorr_crit','off','line_crit','off');

                % Plot (DEBUG ONLY)
                % vis_artifacts(EEG_artif,ASR);title('Contaminated vs ASR')
                data.data{1, run(j)}.X = double(ASR.data)';
                data.data{1, run(j)}.trial = data.data{1, run(j)}.trial - data.data{1, run(j)}.trial(1);
            catch ME
               warning(['artifact removal failed: ', ME.message]);
            end
        end
        
        % actual data extraction
        for i = 1:n(m)
            % index of the trial of interest in a specific run
            tri = data.data{1,run(j)}.trial(trials(m,i));

            % trial start and trial stop
            t_start = floor(tri + tmin*f_samp);
            t_stop  = t_start + n_samp;
            
            for h = 1:k     
                % row index for 'class' and 'signals' related to the
                % run-trial-ch triplet
                index = (j-1)*n(m)*k + (i-1)*k + h;
                
                % class associated to the specific run-trial couple
                class(index) = data.data{1,run(j)}.y(trials(m,i));
                
                % eeg signal during motor imagery associated to the 
                % specific run-trial-ch triplet
                signals(index,:) = data.data{1,run(j)}.X(t_start:t_stop-1, ch(h));
                
                try
                    artifacts(index) = data.data{1, run(j)}.artifacts(trials(m,i));
                catch ME
                    artifacts(index) = 0;
                    if (h == k && i == n(m))
                        warning(['no trial with artifacts removed: ', ME.message]);
                    end
                end
            end
        end
    end

    if (artif == 1)
        signals(artifacts==1,:) =  [];
        class(artifacts==1) = [];
    end
end


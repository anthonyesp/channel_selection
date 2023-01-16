function channel = decision_function(X,n_subjects)
% DECISION_FUNCTION finds the channel leading to the best performance
% according to the mean accuracy and the standard deviation calculated for
% different channel guesses
%
%   INPUT:
%   'X' is a matrix with size 'n_subjects+3' x 'channels number'
%   On the rows there are:
%    - (1) channels found during cross-validation procedure
%    - (2:n_subjects+1) accuracy related to all subjects
%    - (n_subjects+2) mean of accuracy per channel
%    - (n_subjects+3) standard deviation of accuracy per channel
%   On the columns there are the different channels;
%
%   'n_subjects' is the number of subjects.
%
%   OUTPUT:
%   'channel' is the i-th selected channel.
%
%
%  authors:         A. Natalizio
%  correspondence:  anthony.esp@live.it
%  last update:     2020/05/12
	
    ch = X(1,:);
    m = floor(X(n_subjects+2,:));
    s = round(X(n_subjects+3,:));
    
    d = m-s;
    [~,channel] = max(d);
    channel = ch(channel);
end


function [X, freq, time] = STFT(sign,ntrials,nsamp)
%STFT executes the Short Time Fourier Transform on the "ntrials" signals
%     (trials) of length "nsamp" (number of samples)

    % STFT parameters
    nsc = floor(nsamp/6);       % time window samples
    nov = floor(0.9*nsc);       % 90% overlap
    nff = 2^ceil(log2(nsc));    % zero-padding to closest power of 2

    % it deals with NaNs by putting them to zero
    sign(isnan(sign)) = 0;
    
    % Transformation    
    for i = 1:ntrials
        [X(i,:,:), freq, time] = spectrogram(sign(i,:),hamming(nsc),nov,nff);
    end

    % Absolute values => 'transpose' = 'ctranspose' *
    X = abs(X);

end


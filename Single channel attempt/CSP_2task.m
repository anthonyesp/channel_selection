function [W1, W2, C1, C2, n1, n2] = CSP_2task(X1, X2, m)
% CSP creates the projection matrices "Wi", associated to the 2 classes  
% starting from the spectrogram matrices "Xi" separated per class.
%
%   INPUT:
%   'X1' is the 3D array containing only the signals of the class '1';
%   'X2' is the 3D array containing only the signals of the class '2';
%   'm' is the number of first and last rows to consider in the projection
%   matrices 'Wi'.
%
%   OUTPUT:
%   'Wi' is the CSP projection matrix related to class i, with i in range (1,2);
%   'Ci' is the covariance matrix related to class i, with i in range (1,2);
%   'ni' is the trials number to calculate the covariance matrix related to class i, with i in range (1,2);
%
%   NOTE:
%   '*' in the code comments indicated differences with the reference
%   article due to things that are not clear in it.
%
%
%  authors:         A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2020/05/12
    
% Covariance matrices
    % * this calculation is not explicit in the reference article
    nch = size(X1,1);
    
    % Covariance of first class
    n1 = size(X1,3);
    C1 = zeros(nch,nch);
    for i = 1:n1
        M1 = X1(:,:,i)*X1(:,:,i)';
        C1 = C1 + M1/trace(M1);
    end
    C1 = C1/n1;
    
    % Covariance of second class
    n2 = size(X2,3);
    C2 = zeros(nch,nch);
    for i = 1:n2
        M2 = X2(:,:,i)*X2(:,:,i)';
        C2 = C2 + M2/trace(M2);
    end
    C2 = C2/n2;
        
    % Composite Covariance Matrix
    C = C1 + C2;
    
% Projection matrices
    % Complete projection matrices
    [W11, A1] = eig(C1,C);
    [W22, A2] = eig(C2,C);
    
    % Sorting
    [~, ind1] = sort(diag(A1));
    [~, ind2] = sort(diag(A2));

    W111 = W11(:,ind1);
    W222 = W22(:,ind2);

    % Reduced projection matrices
    W1 = [W111(:,1:m), W111(:,end-m+1:end)];
    W2 = [W222(:,1:m), W222(:,end-m+1:end)];
end


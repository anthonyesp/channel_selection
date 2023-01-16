function [WL, WR, WF, WT, CL, CR, CF, CT, nL, nR, nF, nT] = CSP_4task(XL, XR, XF, XT, m)
% CSP creates the projection matrices "Wi", associated to the 4 classes,
% starting from the spectrogram matrices "Xi" separated per class.
%
%   INPUT:
%   'XL' is the 3D array containing only the signals of the class 'left';
%   'XR' is the 3D array containing only the signals of the class 'right';
%   'XF' is the 3D array containing only the signals of the class 'feet';
%   'XT' is the 3D array containing only the signals of the class 'tongue';
%   'm' is the number of first and last rows to consider in the projection
%   matrices 'Wi'.
%
%   OUTPUT:
%   'WL' is the projection matrix associated to the class 'left';
%   'WR' is the projection matrix associated to the class 'right';
%   'WF' is the projection matrix associated to the class 'feet';
%   'WT' is the projection matrix associated to the class 'tongue';
%   'Ci' is the covariance matrix related to class i, with i == (L,R,F,T);
%   'ni' is the trials number to calculate the covariance matrix related to class i, with i == (L,R,F,T);
%
%   NOTE 1:
%   '*' in the code comments indicated differences with the reference
%   article due to things that are not clear in it.
%
%   NOTE 2:
%   at the moment it is only possible to use one channel data!
%
%
%  authors:         A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2020/01/17
    
% Covariance matrices
    % * this calculation is not explicit in the reference article
    nch = size(XL,1);
    
    % Covariance of left class
    nL = size(XL,3);
    CL = zeros(nch,nch);
    for i = 1:nL
        ML = XL(:,:,i)*XL(:,:,i)';
        CL = CL + ML/trace(ML);
    end
    CL = CL/nL;
    
    % Covariance of right class
    nR = size(XR,3);
    CR = zeros(nch,nch);
    for i = 1:nR
        MR = XR(:,:,i)*XR(:,:,i)';
        CR = CR + MR/trace(MR);
    end
    CR = CR/nR;
    
    % Covariance of feet class
    nF = size(XF,3);
    CF = zeros(nch,nch);
    for i = 1:nF
        MF = XF(:,:,i)*XF(:,:,i)';
        CF = CF + MF/trace(MF);
    end
    CF = CF/nF;
    
    % Covariance of toungue class
    nT = size(XT,3);
    CT = zeros(nch,nch);
    for i = 1:nT
        MT = XT(:,:,i)*XT(:,:,i)';
        CT = CT + MT/trace(MT);
    end
    CT = CT/nT;
    
    % Composite Covariance Matrix
    C = CL + CR + CF + CT;
    
% Projection matrices
    % Complete projection matrices
    [W1, A1] = eig(CL,C);
    [W2, A2] = eig(CR,C);
    [W3, A3] = eig(CF,C);
    [W4, A4] = eig(CT,C);
    
    % Sorting
    [~, ind1] = sort(diag(A1));
    [~, ind2] = sort(diag(A2));
    [~, ind3] = sort(diag(A3));
    [~, ind4] = sort(diag(A4));
    
    W1 = W1(:,ind1);
    W2 = W2(:,ind2);
    W3 = W3(:,ind3);
    W4 = W4(:,ind4);

    % Reduced projection matrices
    WL = [W1(:,1:m), W1(:,end-m+1:end)];
    WR = [W2(:,1:m), W2(:,end-m+1:end)];
    WF = [W3(:,1:m), W3(:,end-m+1:end)];
    WT = [W4(:,1:m), W4(:,end-m+1:end)];
end


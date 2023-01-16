function [ind_sel] = MIBIF(F,Y,sel_class,m,k)
% MIBIF Mutual information-based best individual feature (MIBIF)
%
%   INPUT:
%   'F' is the array of all features associated to a specific class;
%   'Y' is the array of classes;
%   'sel_class' is the specific class of interest, for which the best
%   individual features must be identified;
%   'm' is the number of CSP components;
%   'k' is the number of MIBIF components to consider.
%
%   OUTPUT:
%   'ind_sel' are the indexes associated to the best individual features
%   selected because of the higher mutual information between them and the
%   class
%
%
%  authors:         A. Esposito
%  correspondence:  anthony.esp@live.it
%  last update:     2020/05/12

    % one class vs rest
    nt = size(Y,1);
    nc = sum(Y == sel_class);
    nV = size(F,2);
    I = zeros(nV,1);

    % entropy of class vs not class (not clear in the paper*)
    P = nc/nt;
    H = -(P*log2(P) + (1-P)*log2(1-P));
    for j = 1:nV
            
        Hc = 0;
        Hn = 0;
        for i = 1:nt
            % Parzen Window
            % sigma of the distribution
            sig = std(F(i,j) - F(:,j));
            h = sig*(4/(3*nt))^(1/5);
            % Phi
            phi = exp(-((F(i,j) - F(:,j)).^2)/(2*h^2))/sqrt(2*pi);

            % estimation of conditional probability with Parzen Window
            kc = find(Y == sel_class);
            kn = find(Y ~= sel_class);

            cp = sum(phi(kc))/nc;
            np = sum(phi(kn))/(nt - nc);

            % probability of the feature (j,i)
            p = cp*P + np*(1-P);

            % conditional probabilities
            pc = cp*P/p;
            pn = np*(1-P)/p;

            % conditional entropy for 'class' and 'not class'
            Hc = Hc + pc*log2(pc);
            Hn = Hn + pn*log2(pn);
        end

        % Conditional entropy
        Hcond = -(Hc + Hn);
        
        % mutual information of features for desired class
        I(j) = H - Hcond;
    end
    
    % Features selection
    [~, ind] = sort(I);
    
    % selected indexes (first k features in descending order)
    ind_sel = ind(end-(k-1):end);
    
    % complementary indexes (according to CSP)
    if (m == 0)
        ind_com = ind_sel;
    else
        ind_com = (4*m)*ceil(ind_sel/(2*m))-(2*m-1)-ind_sel;
    end
    
    % final indexes
    ind_sel = unique([ind_sel ind_com]);
end


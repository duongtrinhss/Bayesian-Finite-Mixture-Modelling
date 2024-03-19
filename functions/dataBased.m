function [perm] = dataBased(y,K,zmcmc)
% Data-based Algorithm (rodriguez and walker, jcgs)
% y: data
% K: number of component

p = size(y,2);
n = size(y,1);
niter = size(zmcmc,1);

% STEP1: data-based relabelling: finding estimates ("algorithm 5")
cluster_mean = zeros(K,p);
cluster_sd   = zeros(K,p);
cluster_n    = zeros(1,K);

labelvec = (1:K);


% initialize
n_mean = ones(1,K);
n_sd   = ones(1,K);

for j=1:p
    data_r = range(y(:,j));
    data_min = min(y(:,j));
    
    for k=1:K
        cluster_mean(k,j) = data_min + data_r*k/(K+1);
        cluster_sd(k,j)   = data_r/K;
    end
end

% update estimate
perm       = zeros(1,K); % the permutation of st that will be used to reorder the parameter values at each iteration
costmatrix = zeros(K,K); % (K times K) cost matrix of the assignment problem 


sample_mean = zeros(K,p);
sample_sd   = zeros(K,p);

for iter = 1:niter
    alloc = zmcmc(iter,:)';
    for k = 1:K
       ind = find(alloc==k);
       cluster_n(k) = length(ind);       
       for m = 1:K
           costmatrix(k,m) = 0;
           for j=1:p
              costmatrix(k,m) = costmatrix(k,m) + sum(((y(ind,j)-cluster_mean(k,j))/cluster_sd(k,j)).^2);        
           end
       end 
       for j = 1:p
           sample_mean(k,j) = sum(y(ind,j))/cluster_n(k);
           if (cluster_n(k))
               sample_sd(k,j) = sqrt((sum(y(ind,j))-sample_mean(k,j))^2/(cluster_n(k)-1));
           end
       end
    end
    [assign] = solveAssignmentProblem(costmatrix);      % assign[j,i] = 1 <=> index i assigned to index j
    perm = labelvec*(assign>0);                         % the optimal permutation for the current iteration
    
    
    for k = 1:K
        permSum =  cluster_n(perm(k));
        if (permSum>0)
           cluster_mean(k,:) = ((n_mean(k)-1)*cluster_mean(k,:)+sample_mean(perm(k),:))/n_mean(k); 
           n_mean(k) = n_mean(k)+1;
           if (permSum>1)
               cluster_sd(k,:) = ((n_sd(k)-1)*cluster_sd(k,:)+sample_sd(perm(k),:))/n_sd(k);
               n_sd(k) = n_sd(k)+1;
           end
        end
    end
   
end

% STEP2: data-based relabelling: estimates strategy ("algorithm 6")
perm = zeros(niter,K);       % the permutation of st that will be used to reorder the parameter values at each iteration
costmatrix = zeros(K,K); % (K times K) cost matrix of the assignment problem

for iter = 1:niter
    alloc = zmcmc(iter,:)';
    for k = 1:K
        ind = find(alloc==k);
        cluster_n(k) = length(ind);
        for m = 1:K
            costmatrix(k,m) = 0;
            for j=1:p
                costmatrix(k,m) = costmatrix(k,m) + sum(((y(ind,j)-cluster_mean(k,j))/cluster_sd(k,j)).^2);
            end
        end
    end 
    [assign] = solveAssignmentProblem(costmatrix);      % assign[j,i] = 1 <=> index i assigned to index j
    perm(iter,:) = labelvec*(assign > 0);               % the optimal permutation for the current iteration
end

end




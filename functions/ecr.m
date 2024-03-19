function [cfvalue,perm] = ecr(zpivot,zmcmc,K)
% ECR algorithm
% By considering the problem of maxizing the S similarity between zmcmc[,iter] and zpivot
% as a special case of the assignment problem, it minimizes the corresponding cost matrix 
% between zmcmc[,iter] and zpivot
% INPUT: zmcmc: mcmc output of allocation variables with dimension equal to niterxnobs (niter denotes MCMC iterations)
%        zpivot: the pivot

if(K<max(zmcmc,[],'all'))
    error('K should be at least equal to max(z)')
end
if(size(zmcmc,2)~=length(zpivot))
    error("zpivot has not the same length with simulated zmcmc's")
end

labelvec = (1:K);
niter = size(zmcmc,1);
perm = zeros(niter,K);       % the permutation of st that will be used to reorder the parameter values at each iteration
costmatrix = zeros(K,K); % (K times K) cost matrix of the assignment problem

cfvalue = 0; % Initialize value of cost function
for iter = 1:niter
    alloc = zmcmc(iter,:)';
    for j = 1:K
        ind = find(alloc==j);
        so  = zpivot(ind);
        l   = length(ind);
        for m = 1:K
            costmatrix(j,m) = l - length(so(so==j));
        end
    end 
    [assign] = solveAssignmentProblem(costmatrix);      % assign[j,i] = 1 <=> index i assigned to index j
    perm(iter,:) = labelvec*(assign > 0);               % the optimal permutation for the current iteration
    cfvalue = cfvalue + sum(costmatrix.*assign,'all');
end



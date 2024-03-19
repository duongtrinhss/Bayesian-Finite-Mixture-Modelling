function [status,perm] = ecrIterative1(zmcmc,K,threshold,maxiter)
% ecrIterative2: ECR algorithm 1
% INPUT: zmcmc: mcmc output of allocation variables with dimension equal to niterxnobs (niter denotes MCMC iterations)

if(K<max(zmcmc,[],'all'))
    error('K should be at least equal to max(z)')
end

%==========================================================================
%============================| PRELIMINARIES |=============================
labelvec = (1:K);
niter = size(zmcmc,1);
% nobs = size(zmcmc,2);
% perm = zeros(niter,K);       % the permutation of st that will be used to reorder the parameter values at each iteration
% costmatrix = zeros(K,K); % (K times K) cost matrix of the assignment problem

criterion = 99;
if ~exist('threshold','var')
    threshold = 1e-6;
end
if ~exist('maxiter','var')
    maxiter = 100;
end

% =====| Initial permutations: identity
perm = kron(ones(niter,1),labelvec);

% =====| ITERATION 1:
t=1;
% ==========| Estimating pivot

zmcmc_upd = zeros(size(zmcmc));
for iter = 1:niter
    zmcmc_upd(iter,:) = perm(iter,zmcmc(iter,:));
end

zpivot = median(zmcmc_upd,1);

% ==========| Update perm and cost function
[cfvalue,perm] = ecr(zpivot,zmcmc,K);

disp(['This is iteration 1: criterion =', num2str(cfvalue)])  

previous = cfvalue;

%==========================================================================
%=====================| ECR ITERATIONS START HERE |========================
tic;
while ((criterion > threshold)&&(t<maxiter))
    t=t+1;
    % =====| Estimating pivot
    zmcmc_upd = zeros(size(zmcmc));
    for iter = 1:niter
        zmcmc_upd(iter,:) = perm(iter,zmcmc(iter,:));
    end
    
    zpivot = median(zmcmc_upd,1);
    
    % =====| Update perm and cost function
    [cfvalue,perm] = ecr(zpivot,zmcmc,K);
    
    current   = cfvalue;
    criterion = abs(previous-current);
    previous  = cfvalue;
    
    disp(['This is iteration ' num2str(t), ': criterion = ', num2str(cfvalue)]) 
    toc
end

disp(['Iterative ECR algorithm 2 converged at ' num2str(t), ' iterations.'])  

status = ['Converged (', num2str(t),' iterations)' ];
if (criterion> threshold)
    status = 'Max iterations exceeded.';
end

end
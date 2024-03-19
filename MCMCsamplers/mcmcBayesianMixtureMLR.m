function [resFull,resMean,resMedian] = mcmcBayesianMixtureMLR(y,xmat,nsave,nburn)
% mcmcBayesianMixtureMLR

%==========================================================================
%============================| PRELIMINARIES |=============================
if (nargin <= 2)
    nsave = 5000;
    nburn = 500;
end
n     = size(y,1);
k     = size(xmat,2);

% =====| Initialize parameters
ncomp  = 3;
omega  = (1/ncomp)*ones(1,ncomp);
g      = mnrnd(1,omega,n);
alpha  = ones(1,ncomp);

% coefficients
mu_b = zeros(k,1); Q_b = 10*ones(k,1); 
beta = .5*ones(k,ncomp);

% covariances
sigmasq = ones(1,ncomp); s_sig = 3; r_sig = 2;

% =====| Storage space for MCMC

% coefficients
beta_draws    = zeros(nsave+nburn,k,ncomp);

% variances
sigmasq_draws = zeros(nsave+nburn,1,ncomp);

% probability
omega_draws   = zeros(nsave+nburn,1,ncomp);

% component labels
g_draws       = zeros(nsave+nburn,n,ncomp);
prob_draws    = zeros(nsave+nburn,n,ncomp);


%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================
tic;
for iter = 1:nsave+nburn
    
    if mod(iter,100)==0   
        disp(['This is iteration ' num2str(iter)])         
        disp([num2str(100*(iter/(nsave+nburn))) '% completed'])
        toc
    end   
    
    % =====| Draw compnent probability
    % [omega]
    nn                  = sum(g,1);
    omega               = drchrnd(nn + alpha,1);
    
    omega_draws(iter,:) = omega;
    
    % =====| Draw the coefficient vectors
    % [beta]
    
    for l = 1:ncomp
        points   = find(g(:,l)==1);
        if (~isempty(points)==1)
            xg       = xmat(points,:);
            yg       = y(points,:);
            sigmasqg = sigmasq(l);
            
            V_bg  = ( diag(1./Q_b) + xg'*xg/sigmasqg )\eye(k);
            mu_bg = V_bg*( diag(1./Q_b)*mu_b + xg'*yg/sigmasqg );
            
            beta(:,l) = mu_bg + chol(V_bg)'*randn(k,1);   
            
            clear xg yg sigmasqg V_bg mu_bg
        end
    end
    
    beta_draws(iter,:,:) = beta;
   
    % =====| Draw variance 
    % [sigmasq]
    
    for l=1:ncomp
        points = find(g(:,l)==1);
         if (~isempty(points)==1)
            xg       = xmat(points,:);
            yg       = y(points,:);
            betag = beta(:,l);
            
            sse = yg-xg*betag;
            
            s_sigg = s_sig + 0.5*length(points);
            r_sigg = r_sig + 0.5*(sse'*sse);
            
            sigmasq(1,l) = 1./gamrnd(s_sigg,1/r_sigg);
            
            clear xg yg betag s_sigg r_sigg
        end       
    end

    sigmasq_draws(iter,:,:) = sigmasq;
    
    % =====| Draw component latent vectors
    kers = zeros(n,ncomp);
    for l=1:ncomp
        betag = beta(:,l);
        sigmasqg = sigmasq(l);
        omegag = omega(:,l);   
        
        kers(:,l) = log(omegag) + lnormpdf(y-xmat*betag,0,sqrt(sigmasqg));
        
        clear betag sigmasqg omegag
    end
    
    kertrans = exp(kers - max(kers,[],2));
    prob     = kertrans./sum(kertrans,2); 
    g        = mnrnd(1,prob);
    
    prob_draws(iter,:,:) = prob;
    g_draws(iter,:,:) = g;
    
end

%% Save results
beta_save    = beta_draws(nburn+1:nsave+nburn,:,:);
beta1_save   = squeeze(beta_draws(nburn+1:nsave+nburn,1,:));
beta2_save   = squeeze(beta_draws(nburn+1:nsave+nburn,2,:));
sigmasq_save = sigmasq_draws(nburn+1:nburn+nsave,:);
omega_save   = omega_draws(nburn+1:nsave+nburn,:);
g_save       = g_draws(nburn+1:nsave+nburn,:,:);
prob_save    = prob_draws(nburn+1:nsave+nburn,:,:);

resFull.beta_save    = beta_save;
resFull.beta1_save   = beta1_save;
resFull.beta2_save   = beta2_save;
resFull.sigmasq_save = sigmasq_save;
resFull.omega_save   = omega_save;
resFull.g_save       = g_save;
resFull.prob_save    = prob_save;

% Sort and derive posterior mean
resMean.omega_mean_mix = mean(omega_save,1);
[resMean.omega_mean_mix,ind1] = sort(resMean.omega_mean_mix,2,'descend');
resMean.omega_save = omega_save(:,ind1);
resMean.beta1_save = beta1_save(:,ind1);
resMean.beta2_save = beta2_save(:,ind1);
resMean.sigmasq_save = sigmasq_save(:,ind1);
resMean.beta1_mean_mix = reshape(mean(resMean.beta1_save,1),1,ncomp);
resMean.beta2_mean_mix = reshape(mean(resMean.beta2_save,1),1,ncomp);
resMean.sigmasq_mean_mix = reshape(mean(resMean.sigmasq_save,1),1,ncomp);


% Sort and derive posterior median
resMedian.omega_median_mix = median(omega_save,1);
[resMedian.omega_median_mix,ind2] = sort(resMedian.omega_median_mix,2,'descend');
resMedian.omega_save = omega_save(:,ind2);
resMedian.beta1_save = beta1_save(:,ind2);
resMedian.beta2_save = beta2_save(:,ind2);
resMedian.sigmasq_save = sigmasq_save(:,ind2);
resMedian.beta1_median_mix = reshape(median(resMedian.beta1_save,1),1,ncomp);
resMedian.beta2_median_mix = reshape(median(resMedian.beta2_save,1),1,ncomp);
resMedian.sigmasq_median_mix = reshape(median(resMedian.sigmasq_save,1),1,ncomp);
end
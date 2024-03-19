function [resFull,resMean,resMedian] = mcmcBayesianMixture2SLSSSN(Y,D,zmat,xmat,nsave,nburn,g_true)
% mcmcBayesianMixture2SLSSSN
% sample sequentially (SS)
% latent var in selection equation is normalized (N)
% INPUT:
%
% =========================================================================
% Written by Duong Trinh
% University of Glasgow
% This version: July 2023
% =========================================================================

%==========================================================================
%============================| PRELIMINARIES |=============================

if (~exist('nsave','var')||~exist('nburn','var'))
    nsave = 5000;
    nburn = 500;
end

n = size(Y,1);
pz = size(zmat,2);
px = size(xmat,2);
pbar = pz + px;

% =====| Initialize parameters
% mixture params
ncomp  = 3;
omega  = (1/ncomp)*ones(1,ncomp);
g      = mnrnd(1,omega,n);
alpha  = ones(1,ncomp);

% coefficients
mu_gm0 = zeros(pz,1); Q_gm0 = 10^2*ones(pz,1); V_gm0 = diag(Q_gm0);
mu_b0  = zeros(px,1); Q_b0  = 10^2*ones(px,1); V_b0  = diag(Q_b0);
gamma  = 0*ones(pz,1);
beta   = 0*ones(px,ncomp);

% covariances
% for siguv
mu_k0 = 0; Q_k0 = 10^2; V_k0  = diag(Q_k0);
siguv = 0.1*ones(1,ncomp);

% for sigvsq and sigesq
sigvsq = 1; %fixed
c_e0 = 3; d_e0 = 2;

sigesq = ones(1,ncomp);

% =====| Storage space for MCMC
% coefficients
gamma_draws = zeros(nsave+nburn,pz,1);
beta_draws  = zeros(nsave+nburn,px,ncomp);

% covariances
e_v_draws = zeros(nsave+nburn,n,1);
sigesq_draws = zeros(nsave+nburn,1,ncomp);
siguv_draws  = zeros(nsave+nburn,1,ncomp);

% component label vectors
g_draws = zeros(nsave+nburn,n,ncomp);
prob_draws = zeros(nsave+nburn,n,ncomp);

% component probability
omega_draws = zeros(nsave+nburn,ncomp);


%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================

tic;
for iter = 1:nsave+nburn
    
    if mod(iter,100)==0   
        disp(['This is iteration ' num2str(iter)])         
        disp([num2str(100*(iter/(nsave+nburn))) '% completed'])
        toc
    end
    
    % =====| Selection Equation
    % {gamma; v}
    V_gm  = (sigvsq\zmat'*zmat + inv(V_gm0))\eye(pz);
    mu_gm = V_gm*( inv(V_gm0)*mu_gm0 + sigvsq\zmat'*D );  
    gamma = mu_gm + chol(V_gm)'*randn(pz,1);
    gamma_draws(iter,:) = gamma;
        
    e_v = D - zmat*gamma;
    e_v_draws(iter,:,:)   = e_v;
    
    % =====| Outcome Equation    
    % ==========|| Draw compnent probability
    nn                  = sum(g,1);
    omega               = drchrnd(nn + alpha,1);

    omega_draws(iter,:) = omega;
    
    % ==========|| Draw the coefficient vector
     for l = 1:ncomp
         points = find(g(:,l)==1);
         if (~isempty(points))
             x_use = xmat(points,:);
             v_use = e_v(points,:);
             Yg    = Y(points,:);
             
             siguvg   = siguv(:,l);
             sigesqg  = sigesq(:,l);
             
             V_bg  = (sigesqg\x_use'*x_use + inv(V_b0))\eye(px);
             mu_bg = V_bg*( inv(V_b0)*mu_b0 + sigesqg\x_use'*(Yg-siguvg*v_use) );
             
             beta(:,l) = mu_bg + chol(V_bg)'*randn(px,1);
         end
         clear points x_use v_use Yg siguvg sigesqg V_bg mu_bg
     end

    beta_draws(iter,:,:)  = beta;

    % ==========|| Draw the correlation coef
     for l = 1:ncomp
         points = find(g(:,l)==1);
         npts   = sum(g(:,l)==1);
         if (~isempty(points))
             x_use = xmat(points,:);
             v_use = e_v(points,:);
             Yg      = Y(points,:);
             
             betag = beta(:,l);
             sigesqg  = sigesq(:,l);
             siguvg = siguv(:,l);
             
             V_kg  = (sigesqg\v_use'*v_use + inv(V_k0))\1;
             mu_kg  = V_kg*( inv(V_k0)*mu_k0 + sigesqg\v_use'*(Yg-x_use*betag) );
             
             siguv(:,l) = mu_kg + chol(V_kg)'*randn(1,1);
             
         end
         clear points x_use v_use Yg betag siguvg sigesqg V_kg mu_kg sse_Yg
     end

    siguv_draws(iter,:,:)  = siguv;

    % ==========|| Draw the variance
    for l = 1:ncomp
         points = find(g(:,l)==1);
         npts   = sum(g(:,l)==1);
         if (~isempty(points))
             x_use = xmat(points,:);
             v_use = e_v(points,:);
             Yg      = Y(points,:);
             
             betag  = beta(:,l);
             siguvg = siguv(:,l);
    
             sse_Yg = (Yg-x_use*betag-v_use*siguvg)'*(Yg-x_use*betag-v_use*siguvg);
             
             sigesq(:,l) = 1./gamrnd( c_e0+0.5*npts, 1./(d_e0+0.5*sse_Yg) );
         end
         clear points npts x_use v_use Yg betag siguvg sse_Yg
     end

    sigesq_draws(iter,:,:)  = sigesq; 

    % ==========|| Draw component label vectors
    kers = zeros(n,ncomp);
    
    for l = 1:ncomp
        points = find(g(:,l)==1);
        if (~isempty(points))
            omegag = omega(:,l);
            
            gammag = gamma; 
            betag  = beta(:,l);
            siguvg = siguv(:,l);
            sigesqg  = sigesq(:,l);
            
            Sigg = [1 siguvg;
                        siguvg sigesqg^2+siguvg^2];
            Siginvg = inv(Sigg);
   
            for i=1:n
                kers(i,l) = log(omegag) + lnormpdf(Y(i,:)-xmat(i,:)*betag-e_v(i,:)*siguvg,0,sqrt(sigesqg));
            end
            
        end
        clear points omegag betag siguvg sigesqg
    end
     
    kertrans = exp(kers - max(kers,[],2));
    prob = kertrans./sum(kertrans,2);

    g  = mnrnd(1,prob);
    
    prob_draws(iter,:,:) = prob;
    g_draws(iter,:,:) = g;    
end %iter

%% Save results
gamma_save = gamma_draws(nburn+1:nburn+nsave,:);
gamma1_save = gamma_draws(nburn+1:nburn+nsave,1);
gamma2_save = gamma_draws(nburn+1:nburn+nsave,2);
beta_save  = beta_draws(nburn+1:nburn+nsave,:,:);
beta1_save = squeeze(beta_draws(nburn+1:nsave+nburn,1,:));
beta2_save = squeeze(beta_draws(nburn+1:nsave+nburn,2,:));

e_v_save    = e_v_draws(nburn+1:nburn+nsave,:,:);
sigesq_save = sigesq_draws(nburn+1:nburn+nsave,:);
siguv_save  = siguv_draws(nburn+1:nburn+nsave,:);
sigusq_save = sigesq_save + siguv_save.^2;
sigvsq_save = sigvsq*ones(nsave,ncomp);


omega_save = omega_draws(nburn+1:nburn+nsave,:);
g_save       = g_draws(nburn+1:nsave+nburn,:,:);
prob_save    = prob_draws(nburn+1:nsave+nburn,:,:);

resFull.gamma_save = gamma_save;
resFull.gamma1_save = gamma1_save;
resFull.gamma2_save = gamma2_save;
resFull.beta_save = beta_save;
resFull.beta1_save = beta1_save;
resFull.beta2_save = beta2_save;
resFull.sigesq_save = sigesq_save;
resFull.sigusq_save = sigusq_save;
resFull.siguv_save = siguv_save;
resFull.sigvsq_save = sigvsq_save;
resFull.omega_save = omega_save;
resFull.e_v_save = e_v_save;
resFull.g_save = g_save;
resFull.prob_save = prob_save;


% Sort and derive posterior mean
resMean.omega_mean_mix = mean(omega_save,1);
[resMean.omega_mean_mix,ind1] = sort(resMean.omega_mean_mix,2,'descend');
resMean.omega_save = omega_save(:,ind1);
resMean.beta1_save = beta1_save(:,ind1);
resMean.beta2_save = beta2_save(:,ind1);
resMean.sigesq_save = sigesq_save(:,ind1);
resMean.sigusq_save = sigusq_save(:,ind1);
resMean.siguv_save = siguv_save(:,ind1);
resMean.sigvsq_save = sigvsq_save(:,ind1);
resMean.beta1_mean_mix = reshape(mean(resMean.beta1_save,1),1,ncomp);
resMean.beta2_mean_mix = reshape(mean(resMean.beta2_save,1),1,ncomp);
resMean.sigesq_mean_mix  = mean(resMean.sigesq_save,1);
resMean.sigusq_mean_mix  = mean(resMean.sigusq_save,1);
resMean.siguv_mean_mix   = mean(resMean.siguv_save,1);
resMean.sigvsq_mean_mix  = mean(resMean.sigvsq_save,1);

resMean.gamma1_save = gamma1_save;
resMean.gamma1_mean = mean(resMean.gamma1_save,1);
resMean.gamma2_save = gamma2_save;
resMean.gamma2_mean = mean(resMean.gamma2_save,1);

% Sort and derive posterior median
resMedian.omega_median_mix = median(omega_save,1);
[resMedian.omega_median_mix,ind2] = sort(resMedian.omega_median_mix,2,'descend');
resMedian.omega_save = omega_save(:,ind2);
resMedian.beta1_save = beta1_save(:,ind2);
resMedian.beta2_save = beta2_save(:,ind2);
resMedian.sigesq_save = sigesq_save(:,ind2);
resMedian.sigusq_save = sigusq_save(:,ind2);
resMedian.siguv_save = siguv_save(:,ind2);
resMedian.sigvsq_save = sigvsq_save(:,ind2);
resMedian.beta1_median_mix = reshape(median(resMedian.beta1_save,1),1,ncomp);
resMedian.beta2_median_mix = reshape(median(resMedian.beta2_save,1),1,ncomp);
resMedian.sigesq_median_mix  = median(resMedian.sigesq_save,1);
resMedian.sigusq_median_mix  = median(resMedian.sigusq_save,1);
resMedian.siguv_median_mix   = median(resMedian.siguv_save,1);
resMedian.sigvsq_median_mix  = median(resMedian.sigvsq_save,1);

resMedian.gamma1_save = gamma1_save;
resMedian.gamma1_median = median(resMedian.gamma1_save,1);
resMedian.gamma2_save = gamma2_save;
resMedian.gamma2_median = median(resMedian.gamma2_save,1);
end
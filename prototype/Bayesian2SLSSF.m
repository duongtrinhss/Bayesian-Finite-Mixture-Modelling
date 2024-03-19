% Bayesian 2SLS estimation
% =========================================================================             
%
% =========================================================================
% Written by Duong Trinh
% University of Glasgow
% This version: July 2023
% =========================================================================

%==========================================================================
%============================| PRELIMINARIES |=============================
clear all; clc; close all;
ssd = 20230707;
rng(ssd)
addpath(genpath('functions'))

%% DGP
options.form = 3;
n = 5000;

[Y,D,z,x,gamma_true,beta_true,Sigma_true,extout] = genData2SLS(n,options);

zmat = [ones(n,1) z x];
xmat = [ones(n,1) D x];

sigmasq_v_true = Sigma_true(1,1);
sigmasq_u_true = Sigma_true(2,2);
cov_uv_true    = Sigma_true(1,2);

%% Estimation
nsave = 5000;
nburn = 500;
n     = size(Y,1);

pz = size(zmat,2);
px = size(xmat,2);
pbar = pz + px;

pf = 1;

% =====| Initialize parameters
% factors
% loadings
mu_lambD0 = zeros(pf,1); Q_lambD0 = 10^4*ones(pf,1);
mu_lambY0 = zeros(pf,1); Q_lambY0 = 10^4*ones(pf,1);
lambD = 1;
lambY = 1;

% coefficients
mu_gm0 = zeros(pz,1); Q_gm0 = 10^2*ones(pz,1); V_gm0 = diag(Q_gm0);
mu_b0  = zeros(px,1); Q_b0  = 10^2*ones(px,1); V_b0  = diag(Q_b0);
gamma = ones(pz,1);
beta  = ones(px,1);

% variances
c0 = 0.1; d0 = 0.1;
sigvsq = 1; %fixed
sigusq = 1;

% =====| Storage space for MCMC
% factors
f_draws = zeros(nsave+nburn,n,1);

% loadings
lambD_draws = zeros(nsave+nburn,pf);
lambY_draws = zeros(nsave+nburn,pf);

% coefficients
gamma_draws = zeros(nsave+nburn,pz,1);
beta_draws = zeros(nsave+nburn,px,1);

% variances
% sigvsq_draws = zeros(nsave+nburn,1);
sigusq_draws = zeros(nsave+nburn,1);

%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================
tic
for iter = 1: (nsave + nburn)
    
    if mod(iter,100)==0   
        disp(['This is iteration ' num2str(iter)])         
        disp([num2str(100*(iter/(nsave+nburn))) '% completed'])
        toc
    end
    
    % =====| Draw latent factors fi
    Finv = 1+ lambD^2 + lambY^2/sigusq;
    var_f = 1./Finv;

    e_v = D - zmat*gamma;
    e_u = Y - xmat*beta;
    
    mpart = lambD*e_v + lambY/sqrt(sigusq)*e_u;
    mu_f = var_f.*mpart;

    f = mu_f + sqrt(var_f).*randn(n,1);
%     f = extout.f;
    f_draws(iter,:,:) = f;
   
     % =====| Draw coefficient vectors
    coef_D = randn_gibbs_D(D,[zmat f],[Q_gm0' Q_lambD0']',sigvsq,n,pz+pf,1);
    gamma  = coef_D(1:pz);
%     lambD = extout.lambD;
    lambD  = coef_D(pz+1:end);

    coef_Y = randn_gibbs_D(Y,[xmat f],[Q_b0' Q_lambY0']',sigusq,n,px+pf,1);
    beta   = coef_Y(1:px);
%     lambY = extout.lambY;
    lambY  = coef_Y(px+1:end);

    gamma_draws(iter,:,:) = gamma;
    beta_draws(iter,:,:)  = beta;
    lambD_draws(iter,:) = lambD;
    lambY_draws(iter,:) = lambY;

    % =====| Draw covariance matrices
    sse    = sum( (Y-xmat*beta-f*lambY).^2 );
    sigusq = 1./gamrnd( c0+0.5*n+0.5*(px+pf), 1./(d0+0.5*sse) );
    
    sigusq_draws(iter,:) = sigusq;
    
    % =====| Random sign-switch
%     prob = rand(1);
%     sw = (-1)*(prob<=1/2) + 1*(prob>1/2);
% 
%     f     = sw*f;
%     lambD = sw*lambD;
%     lambY = sw*lambY;
end

%% Save results
gamma_save = gamma_draws(nburn+1:nburn+nsave,:,:);
beta_save  = beta_draws(nburn+1:nburn+nsave,:,:);
lambD_save = lambD_draws(nburn+1:nburn+nsave,:);
lambY_save = lambY_draws(nburn+1:nburn+nsave,:);

sigusq_save  = sigusq_draws(nburn+1:nburn+nsave,:);

gamma_mean = mean(gamma_save,1);
beta_mean  = mean(beta_save,1);
lambD_mean = mean(lambD_save,1);
lambY_mean = mean(lambY_save,1);

sigusq_mean = mean(sigusq_save,1);

sigusq_mean + lambY_mean^2
sigvsq + lambD_mean^2

lambY_mean*lambD_mean

f_save = f_draws(nburn+1:nburn+nsave,:,:);
f_mean = mean(f_save,1)';

histogram(f_mean,100)
var(f_mean)


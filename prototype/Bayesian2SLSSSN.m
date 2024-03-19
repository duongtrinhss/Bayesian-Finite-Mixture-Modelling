% Bayesian 2SLS estimation 
% sample sequentially (SS)
% latent var in selection equation is normalized (N)
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
options.form = 4;
% options.R2_d = 0.8;
% options.R2_y = 0.8;
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

% =====| Initialize parameters
% coefficients
mu_gm0 = zeros(pz,1); Q_gm0 = 10^2*ones(pz,1); V_gm0 = diag(Q_gm0);
mu_b0  = zeros(px,1); Q_b0  = 10^2*ones(px,1); V_b0  = diag(Q_b0);

gamma = 0*ones(pz,1);
beta  = 0*ones(px,1);

% covariances
% for siguv
mu_k0 = 0; Q_k0 = 10^2; V_k0  = diag(Q_k0);

siguv  = 0.1;

% for sigvsq and sigesq
c_v0 = 0.1; d_v0 = 0.1;
c_e0 = 0.1; d_e0 = 0.1;

sigvsq = 1;
sigesq = 1;

% =====| Storage space for MCMC
% coefficients
gamma_draws = zeros(nsave+nburn,pz,1);
beta_draws = zeros(nsave+nburn,px,1);

e_v_draws = zeros(nsave+nburn,n,1);
% covariances
sigvsq_draws = zeros(nsave+nburn,1);
sigesq_draws = zeros(nsave+nburn,1);
siguv_draws  = zeros(nsave+nburn,1);

%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================
tic
for iter = 1: (nsave + nburn)
    
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
    gamma_draws(iter,:,:) = gamma;
        
    e_v = D - zmat*gamma;
    e_v_draws(iter,:,:)   = e_v;
    
    
    % =====| Outcome Equation

    V_b  = (sigesq\xmat'*xmat + inv(V_b0))\eye(px);
    mu_b = V_b*( inv(V_b0)*mu_b0 + sigesq\xmat'*(Y-siguv*e_v) );    
    beta = mu_b + chol(V_b)'*randn(px,1);
    beta_draws(iter,:,:)  = beta;
        
    V_k  = (e_v'*e_v/sigesq + inv(V_k0))\1;
    mu_k = V_k*( inv(V_k0)*mu_k0 + e_v'*(Y - xmat*beta)/sigesq );    
    siguv = mu_k + chol(V_k)'*randn(1,1);
      
    sse_Y = (Y-xmat*beta-siguv*e_v)'*(Y-xmat*beta-siguv*e_v);
    sigesq = 1./gamrnd( c_e0+0.5*n, 1./(d_e0+0.5*sse_Y) );
    
    siguv_draws(iter,:)   = siguv;
    sigesq_draws(iter,:)  = sigesq;    
end

%% Save results
gamma_save = gamma_draws(nburn+1:nburn+nsave,:,:);
beta_save  = beta_draws(nburn+1:nburn+nsave,:,:);
e_v_save  = e_v_draws(nburn+1:nburn+nsave,:,:);


sigvsq_save  = sigvsq*ones(nsave,1);  % fixed 
sigesq_save  = sigesq_draws(nburn+1:nburn+nsave,:);
siguv_save   = siguv_draws(nburn+1:nburn+nsave,:);

gamma_mean = mean(gamma_save,1);
beta_mean  = mean(beta_save,1);
e_v_mean  = mean(e_v_save,1);

sigvsq_mean = mean(sigvsq_save,1);
sigesq_mean = mean(sigesq_save,1);
siguv_mean  = mean(siguv_save,1);

disp('True Values, Posterior Means of Parameters');
[sigmasq_v_true sigmasq_u_true cov_uv_true]
[sigvsq_mean sigesq_mean+siguv_mean^2*sigvsq_mean siguv_mean*sigvsq_mean]

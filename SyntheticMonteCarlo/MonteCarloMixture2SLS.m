% Monte Carlo study for Finite Mixture of 2SLS model
% =========================================================================                
%
% =========================================================================
% Written by Duong Trinh
% University of Glasgow
% This version: July 2023
% =========================================================================
clear all; clc; close all;
addpath(genpath('/Users/duongtrinh/Dropbox/FIELDS/Data Science/Matlab/Matlab Practice/BEMtoolbox'))
addpath(genpath('/Users/duongtrinh/Dropbox/FIELDS/Data Science/Matlab/Matlab Practice/SC-HSAR/bayesf_version_2.0'))
addpath(genpath('functions'))
% rng('default');
ssd = 20233107;
rng(ssd)

%==========================================================================
%============================| GENERATE DATA |=============================
options.form = 6;       %==6 mixture & linear model
options.R2_y = 0.50;    %==0.95-highSNR; ==0.75-mediumSNR; ==0.5-lowSNR:
options.heter = 1;      %==0/1/2

n = 2000;
[Y_gen,D_gen,z_gen,x_gen,gamma_true,beta_true,Sigma_true,extout] = genData2SLS(n,options);

sigmasq_v_true = squeeze(Sigma_true(1,1,:));
sigmasq_u_true = squeeze(Sigma_true(2,2,:));
cov_uv_true    = squeeze(Sigma_true(1,2,:));
sigmasq_e_true = sigmasq_u_true - cov_uv_true.^2;
g_true         = extout.g;
omega_true     = extout.omega;

%==========================================================================
%===========================| ESTIMATION SETUP |===========================
%% Choose which model to estimate %% 
est_model = 'BayesianMixture2SLS';
% 'BayesianMixture2SLS'
% 'BayesianMixture2SLSSSN'

%% Choose method for permutation after MCMC sampling %%
perm_method = 'kmeanclustNEW';
% 'aic'
% 'ecr'
% 'ecrIterative1'
% 'ecrIterative2'
% 'kmeanclust'
% 'databased'
% 'kmeanclustNEW'

% list_perm_method = {'aic','kmeanclust','ecr','ecrIterative1','ecrIterative2','databased'};

%% Choose for 'median' or 'mean' for result plotting and comparison based on posteriors%%
opt = 1; %==1 (derive posterior median); ==2 (derive posterior mean)

%% MCMC settings
Y  = Y_gen;
zmat = [ones(n,1) z_gen x_gen];
xmat = [ones(n,1) D_gen x_gen];

nsave = 10000;
nburn = 1000;

%==========================================================================
%==============================| ESTIMATION |==============================
% MCMC SAMPLING
switch est_model
    case 'BayesianMixture2SLS' 
        [resFullmcmc1,resMean1,resMedian1] = mcmcBayesianMixture2SLS(Y_gen,D_gen,zmat,xmat,nsave,nburn,g_true);
        resFullmcmc = resFullmcmc1;
        resMean     = resMean1;
        resMedian   = resMedian1;
    case 'BayesianMixture2SLSSSN' 
        [resFullmcmc2,resMean2,resMedian2] = mcmcBayesianMixture2SLSSSN(Y_gen,D_gen,zmat,xmat,nsave,nburn,g_true);
        resFullmcmc = resFullmcmc2;
        resMean     = resMean2;
        resMedian   = resMedian2;
end

% PERMUTATION

if strcmp(perm_method,'kmeanclustNEW')
   [perm,parperm] = permutemcmcKmeanClustering(resFullmcmc,est_model,opt);
else
   [perm,parperm] = permutemcmcBayesianMixture2SLS(Y,resFullmcmc,est_model,perm_method,opt); 
end

% for i = 1:length(list_perm_method)
%     perm_method = list_perm_method{i};
%     [perm,parperm] = permutemcmcBayesianMixture2SLS(Y,resFullmcmc,est_model,perm_method,opt);
%     plotpermmcmc2SLSmix;
% end

% PLOT RESULTS 
if exist('resFullmcmc','var')
    plotmcmc2SLSmix;
end
if exist('parperm','var')
    plotpermmcmc2SLSmix;
end

% SAVE '.mat'
% savedrawsmcmc2SLSmix;



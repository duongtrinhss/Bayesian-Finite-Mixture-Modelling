% Monte Carlo study for Finite Mixture of 2SLS model
% =========================================================================                
%
% =========================================================================
% Written by Duong Trinh
% University of Glasgow
% This version: July 2023
% =========================================================================
clear all; clc; close all;
addpath(genpath('functions'))
addpath(genpath('Truncated Multivariate Student and Normal'))
% rng('default');
ssd = 20233107;
rng(ssd)

%==========================================================================
%============================| GENERATE DATA |=============================
options.form = 8;       %==8 mixture & non-linear model (binary treatment)
n = 5000;
options.heter = 2;      %==0/1/2

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
est_model = 'BayesianMixture2SLSnonlinear';

%% Choose method for permutation after MCMC sampling %%
//perm_method = 'kmeanclust';
//% 'aic'
//% 'ecr'
//% 'ecrIterative1'
//% 'ecrIterative2'
//% 'kmeanclust'
//% 'databased'
//
//list_perm_method = {'aic','kmeanclust','ecr','ecrIterative1','ecrIterative2','databased'};

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
[resFullmcmc,resMean,resMedian] = mcmcBayesianMixture2SLSnonlinear(Y_gen,D_gen,zmat,xmat,nsave,nburn,g_true);

% PLOT RESULTS 
if exist('resFullmcmc','var')
    plotmcmc2SLSmix;
end
//if exist('parperm','var')
//    plotpermmcmc2SLSmix;
//end

% SAVE '.mat'
% savedrawsmcmc2SLSmix;

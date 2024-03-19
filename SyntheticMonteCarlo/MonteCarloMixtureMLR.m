% Monte Carlo study for Finite Mixture of Multiple Regression Models
% =========================================================================                
%
% =========================================================================
% Written by Duong Trinh
% University of Glasgow
% This version: June 2023
% =========================================================================
clear all; clc; close all;
addpath(genpath('/Users/duongtrinh/Dropbox/FIELDS/Data Science/Matlab/Matlab Practice/BEMtoolbox'))
addpath(genpath('functions'))
addpath(genpath('/Users/duongtrinh/Dropbox/FIELDS/Data Science/Matlab/Matlab Practice/SC-HSAR/bayesf_version_2.0'))
rng('default');

%==========================================================================
%==============================| REFERENCES |==============================
% Function to run MCMC for a finite mixture distribution
open mixturemcmc.m
% Demo
open demo_mixreg.m
open mcmcplot.m
% Sampling beta,sigmasq and allocations
open posterior.m
open prodnormultsim.m
open dataclass.m
% Permutation
open ranpermute.m

%==========================================================================
%============================| GENERATE DATA |=============================
%% DGP setting %%
N = 5000;
k = 2;
optionDGP.SNR = 0; %==1(high SNR), ==0(low SNR)

[Y_gen,X_gen,g_gen,beta_true,sigmasq_true,omega_true,R_square] = genDataMLR(N,k,optionDGP);

%==========================================================================
%==========================| MODEL TO ESTIMATE |===========================
%% Fruhwirth toolbox %%
dataMLR.X = X_gen';
dataMLR.y = Y_gen';
dataMLR.N = N;
dataMLR.sim = 1;

clear mixreg;
mixreg.dist='Normal';
mixreg.d=size(dataMLR.X,1);
mixreg.K=3;

[dataMLR,mixreg,mcmc]=mcmcstart(dataMLR,mixreg);

clear prior;
prior=priordefine(dataMLR,mixreg);

clear mcmcout;
mcmcout=mixturemcmc(dataMLR,mixreg,prior,mcmc);

mean(mcmcout.par.beta,1)
mean(mcmcout.par.sigma,1)


mcmcplot(mcmcout);
[est,mcmcout]=mcmcestimate(mcmcout);

mcmcout=mcmcpermute(mcmcout);
mcmcout.weight

mean(mcmcout.parperm.beta,1)
mean(mcmcout.parperm.sigma,1)

% parclust=zeros(mcmcout.M,2,mcmcout.model.K);
% parclust(:,1,:)=mcmcout.par.mu;
% parclust(:,2,:)=mcmcout.par.sigma.^.5;

estpm=mcmcpm(mcmcout);
indexp=[1:mcmcout.M];

parclust=mcmcout.par.beta;
parclust=permute(parclust,[3 1 2]); %permute such that first index is equal to group, second index to MCMC iteratio, third index to the components of group specific parameter
parclust=reshape(parclust,size(parclust,1)*size(parclust,2),size(parclust,3));
clustart=[estpm.par.beta'];

% [S,clu]=kmeans(parclust,mcmcout.model.K,'EmptyAction','singleton');
[S,clu]=kmeans(parclust,mcmcout.model.K,'start',clustart,'EmptyAction','singleton');
clear parclust;
mcmcout.perm=reshape(S,mcmcout.model.K,size(S,1)/mcmcout.model.K)'; % reshape M times K

mcmcout.isperm=all(sort(mcmcout.perm,2)==repmat([1:mcmcout.model.K],mcmcout.M,1),2);
mcmcout.nonperm=sum(~mcmcout.isperm); % number of draws that may not be permuted uniquely
indexperm=indexp(mcmcout.isperm);  % consider only permuted draws

if isfield(mcmcout.model,'d') d=mcmcout.model.d; else  d=1; end
if isfield(mcmcout.model,'indexdf')
    df=size(mcmcout.model.indexdf,1)*size(mcmcout.model.indexdf,2);
else
    df=0;
end

Mperm=mcmcout.M-mcmcout.nonperm;
mcmcout.Mperm=Mperm;


if size(indexperm,2)>1
    mcmcsub=mcmcsubseq(mcmcout,indexperm);
    mcmcout.parperm=mcmcsub.par; 
    for k=1:mcmcout.model.K
        ik=(mcmcout.perm(indexperm,:)==k*ones(Mperm,mcmcout.model.K,1));
        mcmcout.parperm.sigma(:,k)=sum(ik.*mcmcsub.par.sigma,2);
        ikr(:,1,:)=ik; clear ik;
        mcmcout.parperm.beta(:,:,k)=sum(repmat(ikr,[1 (d-df) 1]).*mcmcsub.par.beta,3);
    end
end

mean(mcmcout.parperm.sigma,1)
mean(mcmcout.parperm.beta,1)

% est.invariant.par.beta
% est.invariant.par.sigma
% est.ident.par.beta
% est.ident.par.sigma
% est.ident.weight
% est.pm.par.beta
% est.ml
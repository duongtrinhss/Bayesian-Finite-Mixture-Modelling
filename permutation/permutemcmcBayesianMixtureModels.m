clear all; clc; close all;
ssd = 20232907;
addpath(genpath('functions'));

load('data/mcmc_output.mat');

zmcmc = mcmc_output.z;
% m     = mcmc_output.m; 
m     = 300;
K     = mcmc_output.K;
J     = mcmc_output.J;
p     = mcmc_output.p;
x     = mcmc_output.x;
mcmc_pars = reshape(mcmc_output.mcmc_pars,[300 K J]);
mapindex = mcmc_output.mapindex;
mapindex_non = mcmc_output.mapindex_non;

zpivot = median(zmcmc,1);

% zpivot2 = zmcmc(mapindex,:);
prapivot = mcmc_pars(mapindex,:,:);

perm_method = 'dataBased';

if (strcmp(perm_method,'ecr')||strcmp(perm_method,'ecrIterative1')||strcmp(perm_method,'ecrIterative2')||strcmp(perm_method,'databased'))    
    
    threshold = 1e-6;
    maxiter = 100;
end

% Permutations

switch perm_method
    case 'aic'
        param = mcmc_pars(:,:,1);
        perm = aic(param,K);
    case 'ecr'
        [~,perm] = ecr(zpivot,zmcmc,K);
    case 'ecrIterative1'
        [~,perm] = ecrIterative1(zmcmc,K,threshold,maxiter);
    case 'ecrIterative2'
        [~,perm] = ecrIterative2(zmcmc,K,p,threshold,maxiter);
    case 'kmeanclust'
        param = mcmc_pars(:,:,1);
        parclust = permute(param,[1,3,2]);
        perm = kmeanclust(parclust,K);
    case 'dataBased'
        [perm] = dataBased(x,K,zmcmc);
end

clear parperm
parperm.parm1 = mcmc_pars(:,:,1);
parperm.parm2 = mcmc_pars(:,:,2);
parperm.parm3 = mcmc_pars(:,:,3);
for l = 1:K
    il = (perm==l*ones(m,K));
    parperm.parm1(:,l) = sum(il.*mcmc_pars(:,:,1),2);
    parperm.parm2(:,l) = sum(il.*mcmc_pars(:,:,2),2);
    parperm.parm3(:,l) = sum(il.*mcmc_pars(:,:,3),2);
end



%==== Parameter 1
figure
subplot(2,1,1)
plot(1:m,mcmc_pars(:,1,1));
subplot(2,1,2)
plot(1:m,mcmc_pars(:,2,1));

figure
plot(1:m,mcmc_pars(:,1,1));
hold on;
plot(1:m,mcmc_pars(:,2,1));

%========| After permutation
figure
plot(1:m,parperm.parm1(:,1));
hold on;
plot(1:m,parperm.parm1(:,2));
xlabel(['Method: ', perm_method]);


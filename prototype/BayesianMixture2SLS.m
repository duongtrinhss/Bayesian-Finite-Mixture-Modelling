% Bayesian mixture 2SLS estimation
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
addpath(genpath('/Users/duongtrinh/Dropbox/FIELDS/Data Science/Matlab/Matlab Practice/BEMtoolbox'))
addpath(genpath('/Users/duongtrinh/Dropbox/FIELDS/Data Science/Matlab/Matlab Practice/SC-HSAR/bayesf_version_2.0'))
addpath(genpath('functions'))
rng('default');

%% DGP setting %%
% options.form = 2; %mixture
% options.R2_d = 0.99;
% options.R2_y = 0.99;
% n = 5000;
% 
% [Y,D,z,x,gamma_true,beta_true,Sigma_true,extout] = genData2SLS(n,options);
% 
% sigmasq_v_true = squeeze(Sigma_true(1,1,:));
% sigmasq_u_true = squeeze(Sigma_true(2,2,:));
% cov_uv_true    = squeeze(Sigma_true(1,2,:));
% 
% g_true     = extout.g;
% omega_true = extout.omega;

% options.form = 5; %mixture
% n = 5000;
% [Y,D,z,x,gamma_true,beta_true,Sigma_true,extout] = genData2SLS(n,options);
% 
% sigmasq_v_true = squeeze(Sigma_true(1,1,:));
% sigmasq_u_true = squeeze(Sigma_true(2,2,:));
% cov_uv_true    = squeeze(Sigma_true(1,2,:));
% sigmasq_e_true = sigmasq_u_true - cov_uv_true.^2;
% g_true = extout.g;
% omega_true = extout.omega;

% options.form = 6; %mixture
% % options.R2_y = 0.75;
% options.heter = 1;
% n = 2000;
% [Y,D,z,x,gamma_true,beta_true,Sigma_true,extout] = genData2SLS(n,options);
% 
% sigmasq_v_true = squeeze(Sigma_true(1,1,:));
% sigmasq_u_true = squeeze(Sigma_true(2,2,:));
% cov_uv_true    = squeeze(Sigma_true(1,2,:));
% sigmasq_e_true = sigmasq_u_true - cov_uv_true.^2;
% g_true = extout.g;
% omega_true = extout.omega;
% 
% zmat = [ones(n,1) z x];
% xmat = [ones(n,1) D x];

options.form = 8;
n = 5000;
options.heter = 2;
[Y,D,z,x,gamma_true,beta_true,Sigma_true,extout] = genData2SLS(n,options);

D = extout.Dstar;

zmat = [ones(n,1) z x];
xmat = [ones(n,1) D x];

sigmasq_v_true = squeeze(Sigma_true(1,1,:));
sigmasq_u_true = squeeze(Sigma_true(2,2,:));
cov_uv_true    = squeeze(Sigma_true(1,2,:));
sigmasq_e_true = sigmasq_u_true - cov_uv_true.^2;
g_true         = extout.g;
omega_true     = extout.omega;
epsD_true      = extout.epsD;

%% ESTIMATIONS
nsave = 5000;
nburn = 500;

% =====| Choose method for permutation after MCMC sampling
perm_method = 'kmeanclust';

% perm_method = 'ecrIterative2';
% 'aic'
% 'ecr'
% 'ecrIterative1'
% 'ecrIterative2'
% 'kmeanclust'
opt = 1; %==1 (derive posterior median); ==2 (derive posterior mean)

% mcmc sampling
[resFullmcmc,resMean,resMedian] = mcmcBayesianMixture2SLS(Y,D,zmat,xmat,nsave,nburn,g_true);
% permutation
[perm,parperm] = permutemcmcBayesianMixture2SLS(resFullmcmc,perm_method,opt);

%% PROTOTYPE
%==========================================================================
%============================| PRELIMINARIES |=============================nsave = 5000;
nsave = 5000;
nburn = 500;
n     = size(Y,1);

pz = size(zmat,2);
px = size(xmat,2);
pbar = pz + px;

% =====| Initialize parameters
% mixture params
ncomp = 3;
omega  = (1/ncomp)*ones(1,ncomp);
g      = mnrnd(1,omega,n);
alpha  = ones(1,ncomp);

% coefficients
mu_gm = zeros(pz,ncomp); Q_gm = 10^2*ones(pz,ncomp); V_gm = repmat(diag(10^2*ones(pz,1)),[1 1 ncomp]);
mu_b  = zeros(px,ncomp); Q_b  = 10^2*ones(px,ncomp); V_b  = repmat(diag(10^2*ones(px,1)),[1 1 ncomp]);

gamma = 0*ones(pz,ncomp);
beta  = 0*ones(px,ncomp);

%==== stack to one vector
mu_th = [mu_gm(:,1); mu_b(:,1)];
V_th  = blkdiag(V_gm(:,:,1),V_b(:,:,1));

% covariances
rho = 4;
R   = eye(2);

Sig = repmat(eye(2), [1 1 ncomp]);
Siginv = zeros(2,2,ncomp);

sigv  = zeros(1,ncomp);
sigu  = zeros(1,ncomp);
siguv = zeros(1,ncomp);

for l=1:ncomp
    Siginv(:,:,l) = inv(Sig(:,:,l));
    sigv(:,l)  = sqrt(Sig(1,1,l));
    sigu(:,l)  = sqrt(Sig(2,2,l));
    siguv(:,l) = Sig(1,2,l);
end 

% =====| Storage space for MCMC
% coefficients
gamma_draws = zeros(nsave+nburn,pz,ncomp);
beta_draws  = zeros(nsave+nburn,px,ncomp);

% covariances
sigv_draws  = zeros(nsave+nburn,ncomp);
sigu_draws  = zeros(nsave+nburn,ncomp);
siguv_draws = zeros(nsave+nburn,ncomp);

% component label vectors
g_draws = zeros(nsave+nburn,n,ncomp);

% component probability
omega_draws = zeros(nsave+nburn,ncomp);

% for convenience
ZZ = zmat'*zmat;
XX = xmat'*xmat;
ZX = zmat'*xmat;
XZ = xmat'*zmat;

%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================

tic;
for iter = 1:nsave+nburn
    
    if mod(iter,100)==0   
        disp(['This is iteration ' num2str(iter)])         
        disp([num2str(100*(iter/(nsave+nburn))) '% completed'])
        toc
    end
    
    % =====| STEP 5    
    % =====| Draw compnent probability
    nn                  = sum(g,1);
    omega               = drchrnd(nn + alpha,1);
%     [omega, ind]        = sort(omega, 'descend');
%     g                   = g(:,ind);
%     omega = omega_true';    
    omega_draws(iter,:) = omega;
    
    % =====| STEP 1
    % =====| Draw the coefficient vectors
    % {gamma; beta}
    
    for l = 1:ncomp
         points = find(g(:,l)==1);
         if (~isempty(points))
             x_use = xmat(points,:);
             z_use = zmat(points,:);
             
             Yg      = Y(points,:);
             Dg      = D(points,:);
             
             Siginvg = Siginv(:,:,l);
             
             ZZ = z_use'*z_use;
             ZX = z_use'*x_use;
             XX = x_use'*x_use;
             XZ = x_use'*z_use;

             s11 = Siginvg(1,1);  s12 = Siginvg(1,2);
             s21 = Siginvg(2,1);  s22 = Siginvg(2,2);
             
             zmata = [(z_use'*s11) (z_use'*s12)];
             xmata = [(x_use'*s21) (x_use'*s22)];
             
             wwpart = [(ZZ*s11) (ZX*s12);...
                 (XZ*s21) (XX*s22) ];
             
             rvecg = [Dg' Yg']';
             
             wrpart = [zmata*rvecg; xmata*rvecg];
             
             covmatpart = (wwpart + inv(V_th))\eye(pbar);
             meanpart   = covmatpart*(wrpart + V_th\mu_th);
             
             theta = meanpart + (chol(covmatpart))'*randn((pbar),1);
             
             gamma(:,l) = theta(1:pz,1);
             beta(:,l)  = theta(pz+1:pbar,1);          
         end
         clear points x_use z_use Dg yg
    end

%     gamma = gamma_true;
%     beta  = beta_true;

    gamma_draws(iter,:,:) = gamma;
    beta_draws(iter,:,:)  = beta;
    
    % =====| Draw covariance matrix
    % {sigv; sigu; siguv}
    
    for l = 1:ncomp
        points = find(g(:,l)==1);
        npts   = sum(g(:,l)==1);
        if (~isempty(points))
            x_use  = xmat(points,:);
            z_use  = zmat(points,:);
            Yg     = Y(points,:);
            Dg     = D(points,:);
            
            gammag = gamma(:,l);
            betag  = beta(:,l);
                 
            e_v = Dg - z_use*gammag;
            e_u = Yg - x_use*betag;           
            
            M = [ (e_v'*e_v) (e_v'*e_u);
                (e_u'*e_v) (e_u'*e_u)];
      
            Sigg = iwishrnd(M + rho*R, npts + rho);
            
            Sig(:,:,l) = Sigg;
            Siginv(:,:,l) = inv(Sigg);
            
            sigv(:,l)  = sqrt(Sigg(1,1));
            sigu(:,l)  = sqrt(Sigg(2,2));
            siguv(:,l) = Sigg(2,1);
        end
        clear points x_use z_use Dg Yg Sigg
    end

%     Sig = Sigma_true;
%     for l=1:ncomp
%         Siginv(:,:,l) = inv(Sig(:,:,l));
%         sigv(:,l)  = sqrt(Sig(1,1,l));
%         sigu(:,l)  = sqrt(Sig(2,2,l));
%         siguv(:,l) = Sig(1,2,l);
%     end
    
    sigv_draws(iter,:)  = sigv;
    sigu_draws(iter,:)  = sigu;
    siguv_draws(iter,:) = siguv;

    % =====| Draw component label vectors
    kers = zeros(n,ncomp);
    
    for l=1:ncomp
        gammag = gamma(:,l);
        betag  = beta(:,l);
        
        Sigg    = Sig(:,:,l);
        Siginvg = Siginv(:,:,l);
        omegag  = omega(:,l);
          
        for i=1:n
            ri = [D(i,:) Y(i,:)]';
            mu_ri  = [(zmat(i,:)*gammag)' (xmat(i,:)*betag)']';
            kers(i,l) = log(omegag)-1/2*log(det(Sigg))-1/2*(ri-mu_ri)'*Siginvg*(ri-mu_ri);
        end
    end
    
    kertrans = exp(kers - max(kers,[],2));
    prob = kertrans./sum(kertrans,2);

    g                 = mnrnd(1,prob);
%     g = g_true;
    g_draws(iter,:,:) = g;
    
end %iter
    
%% Save results
gamma_save = gamma_draws(nburn+1:nburn+nsave,:,:);
gamma_mean_mix = mean(gamma_save,1);
beta_save  = beta_draws(nburn+1:nburn+nsave,:,:);
beta_mean_mix  = mean(beta_save,1);

gamma1_save = squeeze(gamma_draws(nburn+1:nsave+nburn,1,:));
gamma1_mean_mix = reshape(mean(gamma1_save,1),1,ncomp);
gamma2_save = squeeze(gamma_draws(nburn+1:nsave+nburn,2,:));
gamma2_mean_mix = reshape(mean(gamma2_save,1),1,ncomp);

beta1_save = squeeze(beta_draws(nburn+1:nsave+nburn,1,:));
beta1_mean_mix = reshape(mean(beta1_save,1),1,ncomp);
beta2_save = squeeze(beta_draws(nburn+1:nsave+nburn,2,:));
beta2_mean_mix = reshape(mean(beta2_save,1),1,ncomp);


sigv_save  = sigv_draws(nburn+1:nburn+nsave,:);
sigu_save  = sigu_draws(nburn+1:nburn+nsave,:);
siguv_save = siguv_draws(nburn+1:nburn+nsave,:);

sigmasq_v_save = sigv_save.^2;
sigmasq_u_save = sigu_save.^2;
cov_uv_save    = siguv_save;
sigmasq_v_mean_mix  = mean(sigmasq_v_save,1);
sigmasq_u_mean_mix  = mean(sigmasq_u_save,1);
cov_uv_mean_mix     = mean(cov_uv_save,1);

omega_save = omega_draws(nburn+1:nburn+nsave,:);
omega_mean_mix = mean(omega_save,1);

%% Relabeling
% beta2_save_temp = reshape(beta2_save,nsave,[],ncomp);
res_save = [beta_save];
% res_save = [beta2_save_temp];
parclust = res_save;
parclust=permute(parclust,[3 1 2]); %permute such that first index is equal to group, second index to MCMC iteratio, third index to the components of group specific parameter
parclust=reshape(parclust,size(parclust,1)*size(parclust,2),size(parclust,3));
[S,clu]=kmeans(parclust,ncomp,'EmptyAction','singleton');
% [S,clu]=kmeans(parclust,ncomp,'start',clustart,'EmptyAction','singleton');

perm=reshape(S,ncomp,size(S,1)/ncomp)'; % reshape nsave times ncomp

parperm.beta1_save = beta1_save;
parperm.beta2_save = beta2_save;
parperm.gamma1_save = gamma1_save;
parperm.gamma2_save = gamma2_save;
parperm.omega_save = omega_save;
parperm.sigmasq_v_save = sigmasq_v_save;
parperm.sigmasq_u_save = sigmasq_u_save;
parperm.cov_uv_save = cov_uv_save;

for l = 1:ncomp
   il = (perm==l*ones(nsave,ncomp));
   parperm.beta1_save(:,l) = sum(il.*beta1_save,2);
   parperm.beta2_save(:,l) = sum(il.*beta2_save,2);
   parperm.gamma1_save(:,l) = sum(il.*gamma1_save,2);
   parperm.gamma2_save(:,l) = sum(il.*gamma2_save,2);
   parperm.omega_save(:,l) = sum(il.*omega_save,2);
   parperm.sigmasq_v_save(:,l) = sum(il.*sigmasq_v_save,2);
   parperm.sigmasq_u_save(:,l) = sum(il.*sigmasq_u_save,2);
   parperm.cov_uv_save(:,l) = sum(il.*cov_uv_save,2);
end

parperm.omega_mean_mix = mean(parperm.omega_save,1);
[parperm.omega_mean_mix,ind] = sort(parperm.omega_mean_mix,'descend');
parperm.omega_save = parperm.omega_save(:,ind);

parperm.beta1_save = parperm.beta1_save(:,ind);
parperm.beta2_save = parperm.beta2_save(:,ind);
parperm.beta1_mean_mix = mean(parperm.beta1_save,1);
parperm.beta2_mean_mix = mean(parperm.beta2_save,1);

parperm.gamma1_save = parperm.gamma1_save(:,ind);
parperm.gamma2_save = parperm.gamma2_save(:,ind);
parperm.gamma1_mean_mix = mean(parperm.gamma1_save,1);
parperm.gamma2_mean_mix = mean(parperm.gamma2_save,1);

parperm.sigmasq_v_save = parperm.sigmasq_v_save(:,ind);
parperm.sigmasq_u_save = parperm.sigmasq_u_save(:,ind);
parperm.cov_uv_save = parperm.cov_uv_save(:,ind);
parperm.sigmasq_v_mean_mix = mean(parperm.sigmasq_v_save,1);
parperm.sigmasq_u_mean_mix = mean(parperm.sigmasq_u_save,1);
parperm.cov_uv_mean_mix = mean(parperm.cov_uv_save,1);

%% Trace plots 
% beta
figure
subplot(3,2,1)
plot(1:nsave,beta1_save(:,1)); hold on; yline(beta_true(1,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta1_mean_mix(1),'Linewidth',2);
ylabel('\beta_{11}, k=1, g=1');xlabel(['iter; true = ', num2str(beta_true(1,1)), ', est = ', num2str(beta1_mean_mix(1))]);xlim([0 nsave]);
subplot(3,2,3)
plot(1:nsave,beta1_save(:,2)); hold on; yline(beta_true(1,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta1_mean_mix(2),'Linewidth',2);
ylabel('\beta_{12}, k=1, g=2');xlabel(['iter; true = ', num2str(beta_true(1,2)), ', est = ', num2str(beta1_mean_mix(2))]);xlim([0 nsave]);
subplot(3,2,5)
plot(1:nsave,beta1_save(:,3)); hold on; yline(beta_true(1,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta1_mean_mix(3),'Linewidth',2);
ylabel('\beta_{13}, k=1, g=3');xlabel(['iter; true = ', num2str(beta_true(1,3)), ', est = ', num2str(beta1_mean_mix(3))]);xlim([0 nsave]);
subplot(3,2,2)
plot(1:nsave,beta2_save(:,1)); hold on; yline(beta_true(2,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta2_mean_mix(1),'Linewidth',2);
ylabel('\beta_{21}, k=2, g=1');xlabel(['iter; true = ', num2str(beta_true(2,1)), ', est = ', num2str(beta2_mean_mix(1))]);xlim([0 nsave]);
subplot(3,2,4)
plot(1:nsave,beta2_save(:,2)); hold on; yline(beta_true(2,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta2_mean_mix(2),'Linewidth',2);
ylabel('\beta_{22}, k=2, g=2');xlabel(['iter; true = ', num2str(beta_true(2,2)), ', est = ', num2str(beta2_mean_mix(2))]);xlim([0 nsave]);
subplot(3,2,6)
plot(1:nsave,beta2_save(:,3)); hold on; yline(beta_true(2,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta2_mean_mix(3),'Linewidth',2);
ylabel('\beta_{23}, k=2, g=3');xlabel(['iter; true = ', num2str(beta_true(2,3)), ', est = ', num2str(beta2_mean_mix(3))]);xlim([0 nsave]);

figure
plot(1:nsave,beta1_save(:,1)); 
hold on; 
plot(1:nsave,beta1_save(:,2)); 
hold on; 
plot(1:nsave,beta1_save(:,3)); 
hold on; 
plot(1:nsave,beta2_save(:,1)); 
hold on; 
plot(1:nsave,beta2_save(:,2)); 
hold on; 
plot(1:nsave,beta2_save(:,3)); 
hold on; 
ylabel('\beta');xlabel('iter');
legend('\beta_{11}','\beta_{12}','\beta_{13}','\beta_{21}','\beta_{22}','\beta_{23}');

% gamma
figure
subplot(3,2,1)
plot(1:nsave,gamma1_save(:,1)); hold on; yline(gamma_true(1,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma1_mean_mix(1),'Linewidth',2);
ylabel('\gamma_{11}, k=1, g=1');xlabel(['iter; true = ', num2str(gamma_true(1,1)), ', est = ', num2str(gamma1_mean_mix(1))]);xlim([0 nsave]);
subplot(3,2,3)
plot(1:nsave,gamma1_save(:,2)); hold on; yline(gamma_true(1,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma1_mean_mix(2),'Linewidth',2);
ylabel('\gamma_{12}, k=1, g=2');xlabel(['iter; true = ', num2str(gamma_true(1,2)), ', est = ', num2str(gamma1_mean_mix(2))]);xlim([0 nsave]);
subplot(3,2,5)
plot(1:nsave,gamma1_save(:,3)); hold on; yline(gamma_true(1,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma1_mean_mix(3),'Linewidth',2);
ylabel('\gamma_{13}, k=1, g=3');xlabel(['iter; true = ', num2str(gamma_true(1,3)), ', est = ', num2str(gamma1_mean_mix(3))]);xlim([0 nsave]);
subplot(3,2,2)
plot(1:nsave,gamma2_save(:,1)); hold on; yline(gamma_true(2,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma2_mean_mix(1),'Linewidth',2);
ylabel('\gamma_{21}, k=2, g=1');xlabel(['iter; true = ', num2str(gamma_true(2,1)), ', est = ', num2str(gamma2_mean_mix(1))]);xlim([0 nsave]);
subplot(3,2,4)
plot(1:nsave,gamma2_save(:,2)); hold on; yline(gamma_true(2,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma2_mean_mix(2),'Linewidth',2);
ylabel('\gamma_{22}, k=2, g=2');xlabel(['iter; true = ', num2str(gamma_true(2,2)), ', est = ', num2str(gamma2_mean_mix(2))]);xlim([0 nsave]);
subplot(3,2,6)
plot(1:nsave,gamma2_save(:,3)); hold on; yline(gamma_true(2,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma2_mean_mix(3),'Linewidth',2);
ylabel('\gamma_{23}, k=2, g=3');xlabel(['iter; true = ', num2str(gamma_true(2,3)), ', est = ', num2str(gamma2_mean_mix(3))]);xlim([0 nsave]);

figure
plot(1:nsave,gamma1_save(:,1)); 
hold on; 
plot(1:nsave,gamma1_save(:,2)); 
hold on; 
plot(1:nsave,gamma1_save(:,3)); 
hold on; 
plot(1:nsave,gamma2_save(:,1)); 
hold on; 
plot(1:nsave,gamma2_save(:,2)); 
hold on; 
plot(1:nsave,gamma2_save(:,3)); 
hold on; 
ylabel('\gamma');xlabel('iter');
legend('\gamma_{11}','\gamma_{12}','\gamma_{13}','\gamma_{21}','\gamma_{22}','\gamma_{23}');

% omega
figure
subplot(3,1,1)
plot(1:nsave,omega_save(:,1)); hold on; yline(omega_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_mean_mix(1),'Linewidth',2);
ylabel('\omega_{1}, g=1');xlabel(['iter; true = ', num2str(omega_true(1)), ', est = ', num2str(omega_mean_mix(1))]);xlim([0 nsave]);
subplot(3,1,2)
plot(1:nsave,omega_save(:,2)); hold on; yline(omega_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_mean_mix(2),'Linewidth',2);
ylabel('\omega_{2}, g=2');xlabel(['iter; true = ', num2str(omega_true(2)), ', est = ', num2str(omega_mean_mix(2))]);xlim([0 nsave]);
subplot(3,1,3)
plot(1:nsave,omega_save(:,3)); hold on; yline(omega_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_mean_mix(3),'Linewidth',2);
ylabel('\omega_{3}, g=3');xlabel(['iter; true = ', num2str(omega_true(3)), ', est = ', num2str(omega_mean_mix(3))]);xlim([0 nsave]);

figure
plot(1:nsave,omega_save(:,1)); 
hold on;
plot(1:nsave,omega_save(:,2)); 
hold on;
plot(1:nsave,omega_save(:,3)); 
ylabel('\omega');xlabel('iter');
legend('\omega_{1}','\omega_{2}','\omega_{3}');

% sigma
figure
subplot(3,2,1)
plot(1:nsave,sigmasq_v_save(:,1)); hold on; yline(sigmasq_v_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_v_mean_mix(1),'Linewidth',2);
ylabel('\sigma^2_v_{1}, g=1');xlabel(['iter; true = ', num2str(sigmasq_v_true(1)), ', est = ', num2str(sigmasq_v_mean_mix(1))]);xlim([0 nsave]);
subplot(3,2,3)
plot(1:nsave,sigmasq_v_save(:,2)); hold on; yline(sigmasq_v_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_v_mean_mix(2),'Linewidth',2);
ylabel('\sigma^2_v_{2}, g=2');xlabel(['iter; true = ', num2str(sigmasq_v_true(2)), ', est = ', num2str(sigmasq_v_mean_mix(2))]);xlim([0 nsave]);
subplot(3,2,5)
plot(1:nsave,sigmasq_v_save(:,3)); hold on; yline(sigmasq_v_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_v_mean_mix(3),'Linewidth',2);
ylabel('\sigma^2_v_{3}, g=3');xlabel(['iter; true = ', num2str(sigmasq_v_true(3)), ', est = ', num2str(sigmasq_v_mean_mix(3))]);xlim([0 nsave]);
subplot(3,2,2)
plot(1:nsave,sigmasq_u_save(:,1)); hold on; yline(sigmasq_u_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_u_mean_mix(1),'Linewidth',2);
ylabel('\sigma^2_u_{1}, g=1');xlabel(['iter; true = ', num2str(sigmasq_u_true(1)), ', est = ', num2str(sigmasq_u_mean_mix(1))]);xlim([0 nsave]);
subplot(3,2,4)
plot(1:nsave,sigmasq_u_save(:,2)); hold on; yline(sigmasq_u_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_u_mean_mix(2),'Linewidth',2);
ylabel('\sigma^2_u_{2}, g=2');xlabel(['iter; true = ', num2str(sigmasq_u_true(2)), ', est = ', num2str(sigmasq_u_mean_mix(2))]);xlim([0 nsave]);
subplot(3,2,6)
plot(1:nsave,sigmasq_u_save(:,3)); hold on; yline(sigmasq_u_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_u_mean_mix(3),'Linewidth',2);
ylabel('\sigma^2_u_{3}, g=3');xlabel(['iter; true = ', num2str(sigmasq_u_true(3)), ', est = ', num2str(sigmasq_u_mean_mix(3))]);xlim([0 nsave]);

figure
plot(1:nsave,sigmasq_u_save(:,1)); 
hold on;
plot(1:nsave,sigmasq_u_save(:,2)); 
hold on;
plot(1:nsave,sigmasq_u_save(:,3)); 
ylabel('\sigma^2_u');xlabel('iter');
legend('\sigma^2_u_{1}','\sigma^2_u_{2}','\sigma^2_u_{3}')

figure
subplot(3,1,1)
plot(1:nsave,cov_uv_save(:,1)); hold on; yline(cov_uv_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(cov_uv_mean_mix(1),'Linewidth',2);
ylabel('\sigma_{uv,1}, g=1');xlabel(['iter; true = ', num2str(cov_uv_true(1)), ', est = ', num2str(cov_uv_mean_mix(1))]);xlim([0 nsave]);
subplot(3,1,2)
plot(1:nsave,cov_uv_save(:,2)); hold on; yline(cov_uv_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(cov_uv_mean_mix(2),'Linewidth',2);
ylabel('\sigma_{uv,2}, g=2');xlabel(['iter; true = ', num2str(cov_uv_true(2)), ', est = ', num2str(cov_uv_mean_mix(2))]);xlim([0 nsave]);
subplot(3,1,3)
plot(1:nsave,cov_uv_save(:,3)); hold on; yline(cov_uv_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(cov_uv_mean_mix(3),'Linewidth',2);
ylabel('\sigma_{uv3}, g=3');xlabel(['iter; true = ', num2str(cov_uv_true(3)), ', est = ', num2str(cov_uv_mean_mix(3))]);xlim([0 nsave]);

figure
plot(1:nsave,cov_uv_save(:,1)); 
hold on;
plot(1:nsave,cov_uv_save(:,2)); 
hold on;
plot(1:nsave,cov_uv_save(:,3)); 
ylabel('\sigma^2_u');xlabel('iter');
legend('\sigma_{uv,1}','\sigma_{uv,2}','\sigma_{uv,3}')


%% Trace plots 
% parperm.beta
figure
subplot(3,2,1)
plot(1:nsave,parperm.beta1_save(:,1)); hold on; yline(beta_true(1,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.beta1_mean_mix(1),'Linewidth',2);
ylabel('\beta_{11}, k=1, g=1');xlabel(['iter; true = ', num2str(beta_true(1,1)), ', est = ', num2str(parperm.beta1_mean_mix(1))]);xlim([0 nsave]);
subplot(3,2,3)
plot(1:nsave,parperm.beta1_save(:,2)); hold on; yline(beta_true(1,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.beta1_mean_mix(2),'Linewidth',2);
ylabel('\beta_{12}, k=1, g=2');xlabel(['iter; true = ', num2str(beta_true(1,2)), ', est = ', num2str(parperm.beta1_mean_mix(2))]);xlim([0 nsave]);
subplot(3,2,5)
plot(1:nsave,parperm.beta1_save(:,3)); hold on; yline(beta_true(1,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.beta1_mean_mix(3),'Linewidth',2);
ylabel('\beta_{13}, k=1, g=3');xlabel(['iter; true = ', num2str(beta_true(1,3)), ', est = ', num2str(parperm.beta1_mean_mix(3))]);xlim([0 nsave]);
subplot(3,2,2)
plot(1:nsave,parperm.beta2_save(:,1)); hold on; yline(beta_true(2,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.beta2_mean_mix(1),'Linewidth',2);
ylabel('\beta_{21}, k=2, g=1');xlabel(['iter; true = ', num2str(beta_true(2,1)), ', est = ', num2str(parperm.beta2_mean_mix(1))]);xlim([0 nsave]);
subplot(3,2,4)
plot(1:nsave,parperm.beta2_save(:,2)); hold on; yline(beta_true(2,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.beta2_mean_mix(2),'Linewidth',2);
ylabel('\beta_{22}, k=2, g=2');xlabel(['iter; true = ', num2str(beta_true(2,2)), ', est = ', num2str(parperm.beta2_mean_mix(2))]);xlim([0 nsave]);
subplot(3,2,6)
plot(1:nsave,parperm.beta2_save(:,3)); hold on; yline(beta_true(2,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.beta2_mean_mix(3),'Linewidth',2);
ylabel('\beta_{23}, k=2, g=3');xlabel(['iter; true = ', num2str(beta_true(2,3)), ', est = ', num2str(parperm.beta2_mean_mix(3))]);xlim([0 nsave]);

figure
plot(1:nsave,parperm.beta1_save(:,1)); 
hold on; 
plot(1:nsave,parperm.beta1_save(:,2)); 
hold on; 
plot(1:nsave,parperm.beta1_save(:,3)); 
hold on; 
plot(1:nsave,parperm.beta2_save(:,1)); 
hold on; 
plot(1:nsave,parperm.beta2_save(:,2)); 
hold on; 
plot(1:nsave,parperm.beta2_save(:,3)); 
hold on; 
ylabel('\beta');xlabel('iter');
legend('\beta_{11}','\beta_{12}','\beta_{13}','\beta_{21}','\beta_{22}','\beta_{23}');

% parperm.gamma
figure
subplot(3,2,1)
plot(1:nsave,parperm.gamma1_save(:,1)); hold on; yline(gamma_true(1,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.gamma1_mean_mix(1),'Linewidth',2);
ylabel('\gamma_{11}, k=1, g=1');xlabel(['iter; true = ', num2str(gamma_true(1,1)), ', est = ', num2str(parperm.gamma1_mean_mix(1))]);xlim([0 nsave]);
subplot(3,2,3)
plot(1:nsave,parperm.gamma1_save(:,2)); hold on; yline(gamma_true(1,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.gamma1_mean_mix(2),'Linewidth',2);
ylabel('\gamma_{12}, k=1, g=2');xlabel(['iter; true = ', num2str(gamma_true(1,2)), ', est = ', num2str(parperm.gamma1_mean_mix(2))]);xlim([0 nsave]);
subplot(3,2,5)
plot(1:nsave,parperm.gamma1_save(:,3)); hold on; yline(gamma_true(1,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.gamma1_mean_mix(3),'Linewidth',2);
ylabel('\gamma_{13}, k=1, g=3');xlabel(['iter; true = ', num2str(gamma_true(1,3)), ', est = ', num2str(parperm.gamma1_mean_mix(3))]);xlim([0 nsave]);
subplot(3,2,2)
plot(1:nsave,parperm.gamma2_save(:,1)); hold on; yline(gamma_true(2,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.gamma2_mean_mix(1),'Linewidth',2);
ylabel('\gamma_{21}, k=2, g=1');xlabel(['iter; true = ', num2str(gamma_true(2,1)), ', est = ', num2str(parperm.gamma2_mean_mix(1))]);xlim([0 nsave]);
subplot(3,2,4)
plot(1:nsave,parperm.gamma2_save(:,2)); hold on; yline(gamma_true(2,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.gamma2_mean_mix(2),'Linewidth',2);
ylabel('\gamma_{22}, k=2, g=2');xlabel(['iter; true = ', num2str(gamma_true(2,2)), ', est = ', num2str(parperm.gamma2_mean_mix(2))]);xlim([0 nsave]);
subplot(3,2,6)
plot(1:nsave,parperm.gamma2_save(:,3)); hold on; yline(gamma_true(2,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.gamma2_mean_mix(3),'Linewidth',2);
ylabel('\gamma_{23}, k=2, g=3');xlabel(['iter; true = ', num2str(gamma_true(2,3)), ', est = ', num2str(parperm.gamma2_mean_mix(3))]);xlim([0 nsave]);

figure
plot(1:nsave,parperm.gamma1_save(:,1)); 
hold on; 
plot(1:nsave,parperm.gamma1_save(:,2)); 
hold on; 
plot(1:nsave,parperm.gamma1_save(:,3)); 
hold on; 
plot(1:nsave,parperm.gamma2_save(:,1)); 
hold on; 
plot(1:nsave,parperm.gamma2_save(:,2)); 
hold on; 
plot(1:nsave,parperm.gamma2_save(:,3)); 
hold on; 
ylabel('\gamma');xlabel('iter');
legend('\gamma_{11}','\gamma_{12}','\gamma_{13}','\gamma_{21}','\gamma_{22}','\gamma_{23}');

% parperm.omega
figure
subplot(3,1,1)
plot(1:nsave,parperm.omega_save(:,1)); hold on; yline(omega_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.omega_mean_mix(1),'Linewidth',2);
ylabel('\omega_{1}, g=1');xlabel(['iter; true = ', num2str(omega_true(1)), ', est = ', num2str(parperm.omega_mean_mix(1))]);xlim([0 nsave]);
subplot(3,1,2)
plot(1:nsave,parperm.omega_save(:,2)); hold on; yline(omega_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.omega_mean_mix(2),'Linewidth',2);
ylabel('\omega_{2}, g=2');xlabel(['iter; true = ', num2str(omega_true(2)), ', est = ', num2str(parperm.omega_mean_mix(2))]);xlim([0 nsave]);
subplot(3,1,3)
plot(1:nsave,parperm.omega_save(:,3)); hold on; yline(omega_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.omega_mean_mix(3),'Linewidth',2);
ylabel('\omega_{3}, g=3');xlabel(['iter; true = ', num2str(omega_true(3)), ', est = ', num2str(parperm.omega_mean_mix(3))]);xlim([0 nsave]);

figure
plot(1:nsave,parperm.omega_save(:,1)); 
hold on;
plot(1:nsave,parperm.omega_save(:,2)); 
hold on;
plot(1:nsave,parperm.omega_save(:,3)); 
ylabel('\omega');xlabel('iter');
legend('\omega_{1}','\omega_{2}','\omega_{3}');

% parperm.sigmasq_u
figure
subplot(3,2,1)
plot(1:nsave,parperm.sigmasq_v_save(:,1)); hold on; yline(sigmasq_v_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmasq_v_mean_mix(1),'Linewidth',2);
ylabel('\sigma^2_v_{1}, g=1');xlabel(['iter; true = ', num2str(sigmasq_v_true(1)), ', est = ', num2str(parperm.sigmasq_v_mean_mix(1))]);xlim([0 nsave]);
subplot(3,2,3)
plot(1:nsave,parperm.sigmasq_v_save(:,2)); hold on; yline(sigmasq_v_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmasq_v_mean_mix(2),'Linewidth',2);
ylabel('\sigma^2_v_{2}, g=2');xlabel(['iter; true = ', num2str(sigmasq_v_true(2)), ', est = ', num2str(parperm.sigmasq_v_mean_mix(2))]);xlim([0 nsave]);
subplot(3,2,5)
plot(1:nsave,parperm.sigmasq_v_save(:,3)); hold on; yline(sigmasq_v_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmasq_v_mean_mix(3),'Linewidth',2);
ylabel('\sigma^2_v_{3}, g=3');xlabel(['iter; true = ', num2str(sigmasq_v_true(3)), ', est = ', num2str(parperm.sigmasq_v_mean_mix(3))]);xlim([0 nsave]);
subplot(3,2,2)
plot(1:nsave,parperm.sigmasq_u_save(:,1)); hold on; yline(sigmasq_u_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmasq_u_mean_mix(1),'Linewidth',2);
ylabel('\sigma^2_u_{1}, g=1');xlabel(['iter; true = ', num2str(sigmasq_u_true(1)), ', est = ', num2str(parperm.sigmasq_u_mean_mix(1))]);xlim([0 nsave]);
subplot(3,2,4)
plot(1:nsave,parperm.sigmasq_u_save(:,2)); hold on; yline(sigmasq_u_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmasq_u_mean_mix(2),'Linewidth',2);
ylabel('\sigma^2_u_{2}, g=2');xlabel(['iter; true = ', num2str(sigmasq_u_true(2)), ', est = ', num2str(parperm.sigmasq_u_mean_mix(2))]);xlim([0 nsave]);
subplot(3,2,6)
plot(1:nsave,parperm.sigmasq_u_save(:,3)); hold on; yline(sigmasq_u_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmasq_u_mean_mix(3),'Linewidth',2);
ylabel('\sigma^2_u_{3}, g=3');xlabel(['iter; true = ', num2str(sigmasq_u_true(3)), ', est = ', num2str(parperm.sigmasq_u_mean_mix(3))]);xlim([0 nsave]);

% parperm.sigma_uv
figure
subplot(3,1,1)
plot(1:nsave,parperm.cov_uv_save(:,1)); hold on; yline(cov_uv_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.cov_uv_mean_mix(1),'Linewidth',2);
ylabel('\sigma_{uv,1}, g=1');xlabel(['iter; true = ', num2str(cov_uv_true(1)), ', est = ', num2str(parperm.cov_uv_mean_mix(1))]);xlim([0 nsave]);
subplot(3,1,2)
plot(1:nsave,parperm.cov_uv_save(:,2)); hold on; yline(cov_uv_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.cov_uv_mean_mix(2),'Linewidth',2);
ylabel('\sigma_{uv,2}, g=2');xlabel(['iter; true = ', num2str(cov_uv_true(2)), ', est = ', num2str(parperm.cov_uv_mean_mix(2))]);xlim([0 nsave]);
subplot(3,1,3)
plot(1:nsave,parperm.cov_uv_save(:,3)); hold on; yline(cov_uv_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.cov_uv_mean_mix(3),'Linewidth',2);
ylabel('\sigma_{uv3}, g=3');xlabel(['iter; true = ', num2str(cov_uv_true(3)), ', est = ', num2str(parperm.cov_uv_mean_mix(3))]);xlim([0 nsave]);

figure
plot(1:nsave,parperm.cov_uv_save(:,1)); 
hold on;
plot(1:nsave,parperm.cov_uv_save(:,2)); 
hold on;
plot(1:nsave,parperm.cov_uv_save(:,3)); 
ylabel('\sigma^2_u');xlabel('iter');
legend('\sigma_{uv,1}','\sigma_{uv,2}','\sigma_{uv,3}')


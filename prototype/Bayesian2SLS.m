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
options.form = 1;
options.R2_d = 0.8;
options.R2_y = 0.8;
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
mu_gm = zeros(pz,1); Q_gm = 10^2*ones(pz,1); V_gm = diag(Q_gm);
mu_b  = zeros(px,1); Q_b  = 10^2*ones(px,1); V_b  = diag(Q_b);

mu_th = [mu_gm; mu_b];
V_th  = blkdiag(V_gm,V_b);


% covariances
rho = 4;
R   = eye(2);

Sigma = eye(2);
Siginv = inv(Sigma);

sigv = sqrt(Sigma(1,1));
sigu = sqrt(Sigma(2,2));
siguv = Sigma(1,2);

% =====| Storage space for MCMC
% coefficients
gamma_draws = zeros(nsave+nburn,pz,1);
beta_draws = zeros(nsave+nburn,px,1);
% covariances
sigv_draws = zeros(nsave+nburn,1);
sigu_draws = zeros(nsave+nburn,1);
siguv_draws = zeros(nsave+nburn,1);

% for convenience
ZZ = zmat'*zmat;
XX = xmat'*xmat;
ZX = zmat'*xmat;
XZ = xmat'*zmat;


%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================
tic
for iter = 1: (nsave + nburn)
    
    if mod(iter,100)==0   
        disp(['This is iteration ' num2str(iter)])         
        disp([num2str(100*(iter/(nsave+nburn))) '% completed'])
        toc
    end
   
    % =====| Draw the coefficient vectors
    % {gamma; beta}
    s11 = Siginv(1,1);  s12 = Siginv(1,2); 
    s21 = Siginv(2,1);  s22 = Siginv(2,2);
    
    zmata = [(zmat'*s11) (zmat'*s12)];
    xmata = [(xmat'*s21) (xmat'*s22)];
    
    wwpart = [(ZZ*s11) (ZX*s12);...
        (XZ*s21) (XX*s22) ];
    
    rvec = [D' Y']'; 
    
    wrpart = [zmata*rvec; xmata*rvec];
    
    covmatpart = (wwpart + inv(V_th))\eye(pbar);
    meanpart   = covmatpart*(wrpart + V_th\mu_th);
    
    theta = meanpart + (chol(covmatpart))'*randn((pbar),1);
    
    gamma = theta(1:pz,1);
    beta  = theta(pz+1:pbar,1);

%     gamma = gamma_true;
%     beta  = beta_true;

    gamma_draws(iter,:,:) = gamma;
    beta_draws(iter,:,:)  = beta;
    
    % =====| Draw covariance matrix
    % {sigv; sigu; siguv}
    e_v = D - zmat*gamma;
    e_u = Y - xmat*beta;
    
    M = [ (e_v'*e_v) (e_v'*e_u);
          (e_u'*e_v) (e_u'*e_u)];
      
    Sigma = iwishrnd(M + rho*R, n + rho);
%     Sigma = Sigma_true;
    Siginv = inv(Sigma);
    
    sigv  = sqrt(Sigma(1,1));
    sigu  = sqrt(Sigma(2,2));
    siguv = Sigma(2,1);
    
    sigv_draws(iter,:)  = sigv;
    sigu_draws(iter,:)  = sigu;
    siguv_draws(iter,:) = siguv;    
end

%% Save results
gamma_save = gamma_draws(nburn+1:nburn+nsave,:,:);
beta_save  = beta_draws(nburn+1:nburn+nsave,:,:);

sigv_save  = sigv_draws(nburn+1:nburn+nsave,:);
sigu_save  = sigu_draws(nburn+1:nburn+nsave,:);
siguv_save = siguv_draws(nburn+1:nburn+nsave,:);

gamma_mean = mean(gamma_save,1);
beta_mean  = mean(beta_save,1);

sigv_mean = mean(sigv_save,1);
sigu_mean = mean(sigu_save,1);
siguv_mean = mean(siguv_save,1);

sigmasq_v_save = sigv_save.^2;
sigmasq_u_save = sigu_save.^2;
cov_uv_save    = siguv_save;
sigmasq_v_mean = mean(sigmasq_v_save,1);
sigmasq_u_mean = mean(sigmasq_u_save,1);
cov_uv_mean    = mean(cov_uv_save,1);

disp('True Values, Posterior Means of Parameters');
[sigmasq_v_true sigmasq_u_true cov_uv_true]
[sigmasq_v_mean sigmasq_u_mean cov_uv_mean]



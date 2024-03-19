% Bayesian Mixture 2SLS estimation
% Nonlinear Treatment Response Models
% Note: Temp3: ???
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
addpath(genpath('Truncated Multivariate Student and Normal'))

%% DGP
options.form = 8;
n = 5000;

[Y,D,z,x,gamma_true,beta_true,Sigma_true,extout] = genData2SLS(n,options);

zmat = [ones(n,1) z x];
xmat = [ones(n,1) D x];

sigmasq_v_true = Sigma_true(1,1);
sigmasq_u_true = Sigma_true(2,2);
cov_uv_true    = Sigma_true(1,2);
g_true         = extout.g;
omega_true     = extout.omega;

%% Estimation
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

% depvars
Dstar = truncnormD(zeros(n,1),1,D);

% coefficients
mu_gm0 = zeros(pz,ncomp); Q_gm0 = 10^2*ones(pz,ncomp); V_gm0 = repmat(diag(10^2*ones(pz,1)),[1 1 ncomp]);
mu_b0  = zeros(px,ncomp); Q_b0  = 10^2*ones(px,ncomp); V_b0  = repmat(diag(10^2*ones(px,1)),[1 1 ncomp]);

gamma = 0*ones(pz,ncomp);
beta  = 0*ones(px,ncomp);

%==== stack to one vector
mu_th0 = [mu_gm0(:,1); mu_b0(:,1)];
V_th0  = blkdiag(V_gm0(:,:,1),V_b0(:,:,1));

% covariances
% for siguv
mu_k0 = 0; Q_k0 = 10^2; V_k0  = diag(Q_k0);
siguv  = 0.1*ones(1,ncomp);

% for sigvsq and sigesq
c_v0 = 0.1; d_v0 = 0.1;
c_e0 = 0.1; d_e0 = 0.1;
sigvsq = 1*ones(1,ncomp);
sigesq = 1*ones(1,ncomp);

Sig    = repmat(eye(2), [1 1 ncomp]);
Siginv = zeros(2,2,ncomp);

% =====| Storage space for MCMC
% coefficients
gamma_draws = zeros(nsave+nburn,pz,ncomp);
beta_draws  = zeros(nsave+nburn,px,ncomp);
Dstar_draws = zeros(nsave+nburn,n,ncomp);

% covariances
sigvsq_draws = zeros(nsave+nburn,1,ncomp);
sigesq_draws = zeros(nsave+nburn,1,ncomp);
siguv_draws  = zeros(nsave+nburn,1,ncomp);

% component label vectors
g_draws = zeros(nsave+nburn,n,ncomp);

% component probability
omega_draws = zeros(nsave+nburn,ncomp);

%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================
tic
for iter = 1: (nsave + nburn)
    
    if mod(iter,100)==0   
        disp(['This is iteration ' num2str(iter)])         
        disp([num2str(100*(iter/(nsave+nburn))) '% completed'])
        toc
    end
    
    % =====| Draw the latent utility
    % D_star
    for l = 1:ncomp
        points = find(g(:,l)==1);
        if (~isempty(points))
            Yg = Y(points);
            Dg = D(points);
            
            z_use = zmat(points,:);
            x_use = xmat(points,:);
            betag  = beta(:,l);
            gammag = gamma(:,l); 
            
            sigesqg = sigesq(:,l);
            siguvg  = siguv(:,l);
            
            mu_dg = z_use*gammag + siguvg/(sigesqg + siguvg^2)*(Yg-x_use*betag);
            V_dg  = 1 - siguvg^2/(sigesqg + siguvg^2);
            
            Dstar(points,:) = truncnormD(mu_dg,V_dg,Dg,'trandn');
        end
        clear points Yg Dg z_use x_use betag gammag mu_dg V_dg
    end
    
    Dstar_draws = Dstar;
    
    % =====| Draw the covariance params
    for l = 1:ncomp
        points = find(g(:,l)==1);
        npts   = sum(g(:,l)==1);
        if (~isempty(points))
            Yg     = Y(points);
            Dstarg = Dstar(points);
            
            z_use = zmat(points,:);
            x_use = xmat(points,:);
            betag  = beta(:,l);
            gammag = gamma(:,l); 
            
            siguvg  = siguv(:,l);
            sigesqg = sigesq(:,l);
            
            e_vg   = Dstarg - z_use*gammag;
            V_kg   = (e_vg'*e_vg/sigesqg + inv(V_k0))\1;
            mu_kg = V_kg*( inv(V_k0)*mu_k0 + e_vg'*(Yg - x_use*betag)/sigesqg );
            siguvg = mu_kg + chol(V_kg)'*randn(1,1);
            
            sse_Yg  = (Yg-x_use*betag-siguvg*e_vg)'*(Yg-x_use*betag-siguvg*e_vg);
            sigesqg = 1./gamrnd( c_e0+0.5*npts, 1./(d_e0+0.5*sse_Yg) );
            
            siguv(:,l)    = siguvg;
            sigesq(:,l)   = sigesqg;
            Sig(:,:,l)  = [sigesqg+siguvg^2 siguvg;
                                siguvg 1];
            
            Siginv(:,:,l) = inv(Sig(:,:,l));
        end
        clear points Yg Dg z_use x_use betag gammag mu_dg V_dg
    end
    
    siguv_draws(iter,:,:)   = siguv;
    sigesq_draws(iter,:,:)  = sigesq;
    
    % =====| Draw the coefficient vectors
    % {gamma; beta}
    for l = 1:ncomp
        points = find(g(:,l)==1);
        npts   = sum(g(:,l)==1);
        if (~isempty(points))
            Yg     = Y(points);
            Dstarg = Dstar(points);
            
            z_use = zmat(points,:);
            x_use = xmat(points,:);
            betag  = beta(:,l);
            gammag = gamma(:,l); 
            
            siguvg  = siguv(:,l);
            sigesqg = sigesq(:,l);
            
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
            
            rvecg = [Dstarg' Yg']';
            
            wrpart = [zmata*rvecg; xmata*rvecg];
            
            covmatpart = (wwpart + inv(V_th0))\eye(pbar);
            meanpart   = covmatpart*(wrpart + V_th0\mu_th0);
            
            theta = meanpart + (chol(covmatpart))'*randn((pbar),1);
            
            gamma(:,l) = theta(1:pz,1);
            beta(:,l)  = theta(pz+1:pbar,1);
         end
        clear points Yg Dg z_use x_use betag gammag siguvg sigesqg Siginvg
    end   

    [beta(2,:), ind]    = sort(beta(2,:), 'descend');
%     omega               = omega(ind);
    g                   = g(:,ind);
    
%     gamma = gamma_true;
%     beta  = beta_true;
    gamma_draws(iter,:,:) = gamma;
    beta_draws(iter,:,:)  = beta;  
    
    % =====| Draw component label vectors
    kers = zeros(n,ncomp);
    
    for l=1:ncomp
        gammag = gamma(:,l);
        betag  = beta(:,l);
        
        Sigg    = Sig(:,:,l);
        Siginvg = Siginv(:,:,l);
        omegag  = omega(:,l);
          
        for i=1:n
            ri        = [Dstar(i,:) Y(i,:)]';
            mu_ri     = [(zmat(i,:)*gammag)' (xmat(i,:)*betag)']';
            kers(i,l) = log(omegag)-1/2*log(det(Sigg))-1/2*(ri-mu_ri)'*Siginvg*(ri-mu_ri);
        end
    end
    
    kertrans = exp(kers - max(kers,[],2));
    prob = kertrans./sum(kertrans,2);

    g                 = mnrnd(1,prob);
%     g = g_true;
    g_draws(iter,:,:) = g;
    
    % =====| Draw component probability
    nn                  = sum(g,1);
    omega               = drchrnd(nn + alpha,1);
    omega_draws(iter,:) = omega;      
     
end %iter

%% Save results
gamma_save = gamma_draws(nburn+1:nburn+nsave,:,:);
gamma_mean_mix = mean(gamma_save,1);
beta_save  = beta_draws(nburn+1:nburn+nsave,:,:);
beta_mean_mix  = mean(beta_save,1);

beta1_save = squeeze(beta_draws(nburn+1:nsave+nburn,1,:));
beta1_mean_mix = reshape(mean(beta1_save,1),1,ncomp);
beta2_save = squeeze(beta_draws(nburn+1:nsave+nburn,2,:));
beta2_mean_mix = reshape(mean(beta2_save,1),1,ncomp);

% e_v_save    = e_v_draws(nburn+1:nburn+nsave,:,:);
sigesq_save = sigesq_draws(nburn+1:nburn+nsave,:);
siguv_save  = siguv_draws(nburn+1:nburn+nsave,:);

sigesq_mean_mix = mean(sigesq_save,1);
siguv_mean_mix  = mean(siguv_save,1);

omega_save = omega_draws(nburn+1:nburn+nsave,:);
omega_mean_mix = mean(omega_save,1);

gamma_save = gamma_draws(nburn+1:nburn+nsave,:,:);
gamma_mean = mean(gamma_save,1);

disp('True Values, Posterior Means of Parameters');
[sigmasq_v_true sigmasq_u_true cov_uv_true]
[sigvsq*ones(ncomp,1) (sigesq_mean_mix+siguv_mean_mix.^2)' siguv_mean_mix']

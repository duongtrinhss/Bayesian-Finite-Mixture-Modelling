% Bayesian 2SLS estimation 
% sample sequentially (SS)
% latent var in selection equation is normalized (N)
% Note: To be continued (check kers)
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
% options.form = 4;
% n = 5000;
% [Y,D,z,x,gamma_true,beta_true,Sigma_true,extout] = genData2SLS(n,options);
% 
% sigmasq_v_true = Sigma_true(1,1);
% sigmasq_u_true = Sigma_true(2,2);
% cov_uv_true    = Sigma_true(1,2);

% options.form = 5; %mixture
% n = 5000;
% [Y,D,z,x,gamma_true,beta_true,Sigma_true,exout] = genData2SLS(n,options);
% 
% sigmasq_v_true = squeeze(Sigma_true(1,1,:));
% sigmasq_u_true = squeeze(Sigma_true(2,2,:));
% cov_uv_true    = squeeze(Sigma_true(1,2,:));
% sigmasq_e_true = sigmasq_u_true - cov_uv_true.^2;
% g_true = exout.g;
% omega_true = exout.omega;

options.form = 6; %mixture
options.heter = 2;
n = 5000;
[Y,D,z,x,gamma_true,beta_true,Sigma_true,extout] = genData2SLS(n,options);

sigmasq_v_true = squeeze(Sigma_true(1,1,:));
sigmasq_u_true = squeeze(Sigma_true(2,2,:));
cov_uv_true    = squeeze(Sigma_true(1,2,:));
sigmasq_e_true = sigmasq_u_true - cov_uv_true.^2;
g_true         = extout.g;
omega_true     = extout.omega;
epsD_true      = extout.epsD;

zmat = [ones(n,1) z x];
xmat = [ones(n,1) D x];

%% Estimation
nsave = 5000;
nburn = 500;
n     = size(Y,1);

pz    = size(zmat,2);
px    = size(xmat,2);
pbar  = pz + px;

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

%     siguv = cov_uv_true';
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

%     sigesq = sigmasq_e_true';
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

    g_draws(iter,:,:) = g; 
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

e_v_save    = e_v_draws(nburn+1:nburn+nsave,:,:);
sigesq_save = sigesq_draws(nburn+1:nburn+nsave,:);
siguv_save  = siguv_draws(nburn+1:nburn+nsave,:);

sigesq_mean_mix = mean(sigesq_save,1);
siguv_mean_mix = mean(siguv_save,1);

omega_save = omega_draws(nburn+1:nburn+nsave,:);
omega_mean_mix = mean(omega_save,1);

gamma_save = gamma_draws(nburn+1:nburn+nsave,:,:);
gamma_mean = mean(gamma_save,1);

disp('True Values, Posterior Means of Parameters');
[sigmasq_v_true sigmasq_u_true cov_uv_true]
[sigvsq*ones(ncomp,1) (sigesq_mean_mix+siguv_mean_mix.^2)' siguv_mean_mix']


% Sort and derive posterior mean
omega_mean_mix = mean(omega_save,1);
[omega_mean_mix,ind1] = sort(omega_mean_mix,2,'descend');
omega_save = omega_save(:,ind1);
beta1_save = beta1_save(:,ind1);
beta2_save = beta2_save(:,ind1);
sigesq_save = sigesq_save(:,ind1);
siguv_save = siguv_save(:,ind1);

beta1_mean_mix = reshape(mean(beta1_save,1),1,ncomp);
beta2_mean_mix = reshape(mean(beta2_save,1),1,ncomp);
sigesq_mean_mix  = mean(sigesq_save,1);
siguv_mean_mix     = mean(siguv_save,1);


%% Trace plots 
% beta
figure
subplot(3,2,1)
plot(1:nsave,beta1_save(:,1)); hold on; yline(beta_true(1,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta1_mean_mix(1),'Linewidth',2);
ylabel('\beta_{11}, k=1, g=1');xlabel(['iter; true = ', num2str(beta_true(1,1)), ', est = ', num2str(beta1_mean_mix(1))])
subplot(3,2,3)
plot(1:nsave,beta1_save(:,2)); hold on; yline(beta_true(1,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta1_mean_mix(2),'Linewidth',2);
ylabel('\beta_{12}, k=1, g=2');xlabel(['iter; true = ', num2str(beta_true(1,2)), ', est = ', num2str(beta1_mean_mix(2))])
subplot(3,2,5)
plot(1:nsave,beta1_save(:,3)); hold on; yline(beta_true(1,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta1_mean_mix(3),'Linewidth',2);
ylabel('\beta_{13}, k=1, g=3');xlabel(['iter; true = ', num2str(beta_true(1,3)), ', est = ', num2str(beta1_mean_mix(3))])
subplot(3,2,2)
plot(1:nsave,beta2_save(:,1)); hold on; yline(beta_true(2,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta2_mean_mix(1),'Linewidth',2);
ylabel('\beta_{21}, k=2, g=1');xlabel(['iter; true = ', num2str(beta_true(2,1)), ', est = ', num2str(beta2_mean_mix(1))])
subplot(3,2,4)
plot(1:nsave,beta2_save(:,2)); hold on; yline(beta_true(2,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta2_mean_mix(2),'Linewidth',2);
ylabel('\beta_{22}, k=2, g=2');xlabel(['iter; true = ', num2str(beta_true(2,2)), ', est = ', num2str(beta2_mean_mix(2))])
subplot(3,2,6)
plot(1:nsave,beta2_save(:,3)); hold on; yline(beta_true(2,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta2_mean_mix(3),'Linewidth',2);
ylabel('\beta_{23}, k=2, g=3');xlabel(['iter; true = ', num2str(beta_true(2,3)), ', est = ', num2str(beta2_mean_mix(3))])

% gamma


% omega
figure
subplot(3,1,1)
plot(1:nsave,omega_save(:,1)); hold on; yline(omega_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_mean_mix(1),'Linewidth',2);
ylabel('\omega_{1}, g=1');xlabel(['iter; true = ', num2str(omega_true(1)), ', est = ', num2str(omega_mean_mix(1))])
subplot(3,1,2)
plot(1:nsave,omega_save(:,2)); hold on; yline(omega_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_mean_mix(2),'Linewidth',2);
ylabel('\omega_{2}, g=2');xlabel(['iter; true = ', num2str(omega_true(2)), ', est = ', num2str(omega_mean_mix(2))])
subplot(3,1,3)
plot(1:nsave,omega_save(:,3)); hold on; yline(omega_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_mean_mix(3),'Linewidth',2);
ylabel('\omega_{3}, g=3');xlabel(['iter; true = ', num2str(omega_true(3)), ', est = ', num2str(omega_mean_mix(3))])

% sigma
figure
subplot(3,2,1)
plot(1:nsave,sigesq_save(:,1)); hold on; yline(sigmasq_e_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigesq_mean_mix(1),'Linewidth',2);
ylabel('\sigma^2_e_{1}, g=1');xlabel(['iter; true = ', num2str(sigmasq_e_true(1)), ', est = ', num2str(sigesq_mean_mix(1))])
subplot(3,2,3)
plot(1:nsave,sigesq_save(:,2)); hold on; yline(sigmasq_e_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigesq_mean_mix(2),'Linewidth',2);
ylabel('\sigma^2_e_{2}, g=2');xlabel(['iter; true = ', num2str(sigmasq_e_true(2)), ', est = ', num2str(sigesq_mean_mix(2))])
subplot(3,2,5)
plot(1:nsave,sigesq_save(:,3)); hold on; yline(sigmasq_e_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigesq_mean_mix(3),'Linewidth',2);
ylabel('\sigma^2_e_{3}, g=3');xlabel(['iter; true = ', num2str(sigmasq_e_true(3)), ', est = ', num2str(sigesq_mean_mix(3))])
subplot(3,2,2)
plot(1:nsave,siguv_save(:,1)); hold on; yline(cov_uv_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(siguv_mean_mix(1),'Linewidth',2);
ylabel('\sigma_{uv,1}, g=1');xlabel(['iter; true = ', num2str(cov_uv_true(1)), ', est = ', num2str(siguv_mean_mix(1))])
subplot(3,2,4)
plot(1:nsave,siguv_save(:,2)); hold on; yline(cov_uv_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(siguv_mean_mix(2),'Linewidth',2);
ylabel('\sigma_{uv,2}, g=2');xlabel(['iter; true = ', num2str(cov_uv_true(2)), ', est = ', num2str(siguv_mean_mix(2))])
subplot(3,2,6)
plot(1:nsave,siguv_save(:,3)); hold on; yline(cov_uv_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(siguv_mean_mix(3),'Linewidth',2);
ylabel('\sigma_{uv,3}, g=3');xlabel(['iter; true = ', num2str(cov_uv_true(3)), ', est = ', num2str(siguv_mean_mix(3))])



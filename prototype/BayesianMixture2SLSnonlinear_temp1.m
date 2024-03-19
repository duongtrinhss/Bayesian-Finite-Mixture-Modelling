% Bayesian Mixture 2SLS estimation
% Nonlinear Treatment Response Models
% Note - Temp1: Not Impose constraint on 'siguv' and 'sigesq'
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
% options.form = 7;
% n = 5000;
% 
% [Y,D,z,x,gamma_true,beta_true,Sigma_true,extout] = genData2SLS(n,options);
% 
% zmat = [ones(n,1) z x];
% xmat = [ones(n,1) D x];
% 
% sigmasq_v_true = Sigma_true(1,1);
% sigmasq_u_true = Sigma_true(2,2);
% cov_uv_true    = Sigma_true(1,2);
% sigmasq_e_true = sigmasq_u_true - cov_uv_true.^2;


options.form = 8;
n = 5000;
options.heter = 2;
[Y,D,z,x,gamma_true,beta_true,Sigma_true,extout] = genData2SLS(n,options);

zmat = [ones(n,1) z x];
xmat = [ones(n,1) D x];

sigmasq_v_true = squeeze(Sigma_true(1,1,:));
sigmasq_u_true = squeeze(Sigma_true(2,2,:));
cov_uv_true    = squeeze(Sigma_true(1,2,:));
sigmasq_e_true = sigmasq_u_true - cov_uv_true.^2;
g_true         = extout.g;
omega_true     = extout.omega;
epsD_true      = extout.epsD;

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
mu_gm0 = zeros(pz,1); Q_gm0 = 10^2*ones(pz,1); V_gm0 = diag(Q_gm0);
mu_b0  = zeros(px,1); Q_b0  = 10^2*ones(px,1); V_b0  = diag(Q_b0);

gamma = 0*ones(pz,ncomp);
beta  = 0*ones(px,ncomp);

%==== stack to one vector
mu_th0 = [mu_gm0; mu_b0];
V_th0  = blkdiag(V_gm0,V_b0);

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
    
    % =====| Draw component probability
    nn                  = sum(g,1);
    omega               = drchrnd(nn + alpha,1);
%     [omega, ind]        = sort(omega, 'descend');
%     g                   = g(:,ind);
%     omega = omega_true';    
    
    omega_draws(iter,:) = omega;
        
    % =====| Draw the covariance params
    % {siguv; sigesq}
    
    for l = 1:ncomp
        points = find(g(:,l)==1);
        npts   = sum(g(:,l)==1);
        if (~isempty(points))
            Yg     = Y(points,:);
            Dstarg = Dstar(points,:);
            
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
            Sig(:,:,l)  = [1 siguvg;
                                siguvg sigesqg+siguvg^2];
            
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

            sigvsqg = sigvsq(:,l);
            siguvg  = siguv(:,l);
            sigesqg = sigesq(:,l);
            
            Siginvg = Siginv(:,:,l);
            
%             ZZ = z_use'*z_use;
%             ZX = z_use'*x_use;
%             XX = x_use'*x_use;
%             XZ = x_use'*z_use;
%             
%             s11 = Siginvg(1,1);  s12 = Siginvg(1,2);
%             s21 = Siginvg(2,1);  s22 = Siginvg(2,2);
%             
%             zmata = [(z_use'*s11) (z_use'*s12)];
%             xmata = [(x_use'*s21) (x_use'*s22)];
%             
%             wwpart = [(ZZ*s11) (ZX*s12);...
%                 (XZ*s21) (XX*s22) ];
%             
%             rvecg = [Dstarg' Yg']';
%             
%             wrpart = [zmata*rvecg; xmata*rvecg];
%             
%             covmatpart = (wwpart + inv(V_th0))\eye(pbar);
%             meanpart   = covmatpart*(wrpart + V_th0\mu_th0);
%             
%             theta = meanpart + (chol(covmatpart))'*randn((pbar),1);
%             
%             gamma(:,l) = theta(1:pz,1);
%             beta(:,l)  = theta(pz+1:pbar,1);
            
            V_gm  = (sigvsqg\z_use'*z_use + inv(V_gm0))\eye(pz);
            mu_gm = V_gm*( inv(V_gm0)*mu_gm0 + sigvsqg\z_use'*Dstarg );
            gamma(:,l) = mu_gm + chol(V_gm)'*randn(pz,1);
            
            v_use = Dstarg - z_use*gamma(:,l);
                        
            V_bg  = (sigesqg\x_use'*x_use + inv(V_b0))\eye(px);
            mu_bg = V_bg*( inv(V_b0)*mu_b0 + sigesqg\x_use'*(Yg-siguvg*v_use) );            
            beta(:,l) = mu_bg + chol(V_bg)'*randn(px,1);

         end
        clear points Yg Dg z_use x_use betag gammag siguvg sigesqg Siginvg
    end  

    gamma_draws(iter,:,:) = gamma;
    beta_draws(iter,:,:)  = beta;  
     
    % =====| Draw the latent utility
    % D_star
    for l = 1:ncomp
        points = find(g(:,l)==1);
        if (~isempty(points))
            Yg = Y(points,:);
            Dg = D(points,:);
            
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
[sigvsq' (sigesq_mean_mix+siguv_mean_mix.^2)' siguv_mean_mix']

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


% Relabeling
parclust=beta_save;
parclust=permute(parclust,[3 1 2]); %permute such that first index is equal to group, second index to MCMC iteratio, third index to the components of group specific parameter
parclust=reshape(parclust,size(parclust,1)*size(parclust,2),size(parclust,3));
[S,clu]=kmeans(parclust,ncomp,'EmptyAction','singleton');
% [S,clu]=kmeans(parclust,ncomp,'start',clustart,'EmptyAction','singleton');

perm=reshape(S,ncomp,size(S,1)/ncomp)'; % reshape nsave times ncomp

parperm.beta1_save = beta1_save;
parperm.beta2_save = beta2_save;
% parperm.sigmasq_save = sigmasq_save;
parperm.omega_save = omega_save;
for l = 1:ncomp
   il = (perm==l*ones(nsave,ncomp));
   parperm.beta1_save(:,l) = sum(il.*beta1_save,2);
   parperm.beta2_save(:,l) = sum(il.*beta2_save,2);
%    parperm.sigmasq_save(:,l) = sum(il.*sigmasq_save,2);
   parperm.omega_save(:,l) = sum(il.*omega_save,2);
end

parperm.omega_mean_mix = mean(parperm.omega_save,1);
[parperm.omega_mean_mix,ind] = sort(parperm.omega_mean_mix,'descend');
parperm.omega_save = parperm.omega_save(:,ind);
parperm.beta1_save = parperm.beta1_save(:,ind);
parperm.beta2_save = parperm.beta2_save(:,ind);
% parperm.sigmasq_save = parperm.sigmasq_save(:,ind);

parperm.beta1_mean_mix = mean(parperm.beta1_save,1);
parperm.beta2_mean_mix = mean(parperm.beta2_save,1);
% parperm.sigmasq_mean_mix = mean(parperm.sigmasq_save,1);


%% Plot
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

%% Trace plots 
% parperm.beta
figure
subplot(3,2,1)
plot(1:nsave,parperm.beta1_save(:,1)); hold on; yline(beta_true(1,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.beta1_mean_mix(1),'Linewidth',2);
ylabel('\beta_{11}, k=1, g=1');xlabel(['iter; true = ', num2str(beta_true(1,1)), ', est = ', num2str(parperm.beta1_mean_mix(1))])
subplot(3,2,3)
plot(1:nsave,parperm.beta1_save(:,2)); hold on; yline(beta_true(1,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.beta1_mean_mix(2),'Linewidth',2);
ylabel('\beta_{12}, k=1, g=2');xlabel(['iter; true = ', num2str(beta_true(1,2)), ', est = ', num2str(parperm.beta1_mean_mix(2))])
subplot(3,2,5)
plot(1:nsave,parperm.beta1_save(:,3)); hold on; yline(beta_true(1,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.beta1_mean_mix(3),'Linewidth',2);
ylabel('\beta_{13}, k=1, g=3');xlabel(['iter; true = ', num2str(beta_true(1,3)), ', est = ', num2str(parperm.beta1_mean_mix(3))])
subplot(3,2,2)
plot(1:nsave,parperm.beta2_save(:,1)); hold on; yline(beta_true(2,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.beta2_mean_mix(1),'Linewidth',2);
ylabel('\beta_{21}, k=2, g=1');xlabel(['iter; true = ', num2str(beta_true(2,1)), ', est = ', num2str(parperm.beta2_mean_mix(1))])
subplot(3,2,4)
plot(1:nsave,parperm.beta2_save(:,2)); hold on; yline(beta_true(2,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.beta2_mean_mix(2),'Linewidth',2);
ylabel('\beta_{22}, k=2, g=2');xlabel(['iter; true = ', num2str(beta_true(2,2)), ', est = ', num2str(parperm.beta2_mean_mix(2))])
subplot(3,2,6)
plot(1:nsave,parperm.beta2_save(:,3)); hold on; yline(beta_true(2,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.beta2_mean_mix(3),'Linewidth',2);
ylabel('\beta_{23}, k=2, g=3');xlabel(['iter; true = ', num2str(beta_true(2,3)), ', est = ', num2str(parperm.beta2_mean_mix(3))])

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

figure
plot(1:nsave,parperm.beta1_save(:,1)); 
hold on; 
plot(1:nsave,parperm.beta1_save(:,2)); 
hold on; 
plot(1:nsave,parperm.beta1_save(:,3)); 
ylabel('\beta');xlabel('iter');
legend('\beta_{11}','\beta_{12}','\beta_{13}');

figure
plot(1:nsave,parperm.beta2_save(:,1)); 
hold on; 
plot(1:nsave,parperm.beta2_save(:,2)); 
hold on; 
plot(1:nsave,parperm.beta2_save(:,3)); 
ylabel('\beta');xlabel('iter');
legend('\beta_{11}','\beta_{12}','\beta_{13}');

% parperm.sigmasq
figure
subplot(3,1,1)
plot(1:nsave,parperm.sigmasq_save(:,1)); hold on; yline(sigmasq_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmasq_mean_mix(1),'Linewidth',2);
ylabel('\sigma^2_{1}, g=1');xlabel(['iter; true = ', num2str(sigmasq_true(1)), ', est = ', num2str(parperm.sigmasq_mean_mix(1))])
subplot(3,1,2)
plot(1:nsave,parperm.sigmasq_save(:,2)); hold on; yline(sigmasq_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmasq_mean_mix(2),'Linewidth',2);
ylabel('\sigma^2_{2}, g=2');xlabel(['iter; true = ', num2str(sigmasq_true(2)), ', est = ', num2str(parperm.sigmasq_mean_mix(2))])
subplot(3,1,3)
plot(1:nsave,parperm.sigmasq_save(:,3)); hold on; yline(sigmasq_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmasq_mean_mix(3),'Linewidth',2);
ylabel('\sigma^2_{3}, g=3');xlabel(['iter; true = ', num2str(sigmasq_true(3)), ', est = ', num2str(parperm.sigmasq_mean_mix(3))])

figure
plot(1:nsave,parperm.sigmasq_save(:,1)); 
hold on;
plot(parperm.sigmasq_save(:,2)); 
hold on;
plot(parperm.sigmasq_save(:,3)); 
ylabel('\sigma^2');xlabel('iter');
legend('\sigma^2_{1}','\sigma^2_{2}','\sigma^2_{3}');

% parperm.omega
figure
subplot(3,1,1)
plot(1:nsave,parperm.omega_save(:,1)); hold on; yline(omega_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.omega_mean_mix(1),'Linewidth',2);
ylabel('\omega_{1}, g=1');xlabel(['iter; true = ', num2str(omega_true(1)), ', est = ', num2str(parperm.omega_mean_mix(1))])
subplot(3,1,2)
plot(1:nsave,parperm.omega_save(:,2)); hold on; yline(omega_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.omega_mean_mix(2),'Linewidth',2);
ylabel('\omega_{2}, g=2');xlabel(['iter; true = ', num2str(omega_true(2)), ', est = ', num2str(parperm.omega_mean_mix(2))])
subplot(3,1,3)
plot(1:nsave,parperm.omega_save(:,3)); hold on; yline(omega_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.omega_mean_mix(3),'Linewidth',2);
ylabel('\omega_{3}, g=3');xlabel(['iter; true = ', num2str(omega_true(3)), ', est = ', num2str(parperm.omega_mean_mix(3))])

figure
plot(1:nsave,parperm.omega_save(:,1)); 
hold on;
plot(1:nsave,parperm.omega_save(:,2)); 
hold on;
plot(1:nsave,parperm.omega_save(:,3)); 
ylabel('\omega');xlabel('iter');
legend('\omega_{1}','\omega_{2}','\omega_{3}');





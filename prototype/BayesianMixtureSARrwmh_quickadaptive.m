% Bayesian estimation of mixture SAR 
% Sampling etwork effect (rho): Random-walk Metropolis-Hasting (quick adaptive algorithm)
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
ssd = 20230307;
rng(ssd)
addpath(genpath('functions'))
addpath(genpath('leSage_toolbox_may2021/'))
addpath(genpath('BEMtoolbox/'))
addpath(genpath('Truncated Multivariate Student and Normal'))

%% DGP setting %%
options.form = 2; % mixture SAR && exogenous spatial setting
N = 2000;
options.R2 = 0.99;
[Y_gen,Wr_gen,X_gen,beta_true,rho_true,sigmasq_true,extout] = genDataSAR(N,options);
g_true = extout.g;
omega_true = extout.omega;


% options.form = 3; % mixture SAR && exogenous network setting
% options.nwdist = 'probit';
% N = 500;
% [Y_gen,Wr_gen,X_gen,beta_true,rho_true,sigmasq_true,extout] = genDataSAR(N,options);

%% Estimation %%
Y  = Y_gen;
X  = X_gen;
Wr = Wr_gen;
N = size(Y,1);
k = size(X,2);

% MCMC sampling setting %
nsave = 2000;
nburn = 200;

% =====| Initialize parameters and value of hyperparameters
ncomp = 3;
omega = (1/ncomp)*ones(1,ncomp);
g     = mnrnd(1,omega,N);
alpha = 2*ones(1,ncomp);

rho     = 0*[-0.3, 0.3, 0.6]; d_rho = 1.01;
beta    = zeros(k,ncomp); mu_b = 0*ones(k,1); Q_b = 10^2*ones(k,1);
sigmav  = ones(1,ncomp); s_sig = 3; r_sig = 2; % shape and scale

% =====| Storage space for MCMC

% store all parameters
rho_draws    = zeros(nsave+nburn,ncomp);        %store rho
beta_draws   = zeros(nsave+nburn,k,ncomp);      %store beta
sigmav_draws = zeros(nsave+nburn,1,ncomp);      %store variance
rho_p_draws  = zeros(nsave+nburn,ncomp);        %store rho

% component lable vectors
g_draws      = zeros(nsave+nburn,N,ncomp);
prob_draws      = zeros(nsave+nburn,N,ncomp);
% component probability
omega_draws  = zeros(nsave+nburn,ncomp);

% tunning in M-H algorithm
c_rho       = 1*ones(nsave+nburn,ncomp);
accrate_rho = zeros(nsave+nburn,ncomp);
acc_rho     = zeros(1,ncomp);

% log-likelihood
logliks = zeros(nsave+nburn,1);

%==========================================================================
%=====================| MCMC ITERATIONS START HERE |=======================
tic;
beta_draws(1,:,:) = beta;
sigmav_draws(1,:) = sigmav;
rho_draws(1,:)    = rho;
g_draws(1,:,:)    = g;
omega_draws(1,:)  = omega;

for iter = 2:nsave+nburn
    
    if mod(iter,100)==0
        disp(['This is iteration ' num2str(iter)])
        disp([num2str(100*(iter/(nsave+nburn))) '% completed'])
        toc
    end
    
    % =====| STEP 5
    % =====| Sample component probability
    % tic;
    % disp(['Sampling component probability...'])
    nn    = sum(g,1);
    omega = drchrnd(nn+alpha,1);
    [omega, ind] = sort(omega,'descend');
    omega_draws(iter,:) = omega;
    g = g(:,ind);
    % toc
    
    % Transform dependent variable
    rho_til = g*rho';
    SS      = speye(N) - diag(rho_til)*Wr;
    Ytilde  = SS*Y;
    
    % =====| STEP 1
    % =====| Sample beta
    % tic;
    % disp(['Sampling beta...'])
    for l=1:ncomp
        points = find(g(:,l)==1);
        if (~isempty(points))
            sigmavg = sigmav(1,l);
            Ytildeg = Ytilde(points,:);
            Xg  = squeeze(X(points,:,:));
            
            V_bg  = ( diag(1./Q_b) + Xg'*Xg/sigmavg )\eye(k);
            mu_bg = V_bg*( diag(1./Q_b)*mu_b + Xg'*Ytildeg/sigmavg );
            
            beta(:,l) = ( mu_bg+norm_rnd(V_bg) );
        end
    end
    
    beta_draws(iter,:,:) = beta;
    % toc
    
    % =====| STEP 2
    % =====| Sample sigmav
    % tic;
    % disp(['Sampling sigmasq...'])
    for l=1:ncomp
        points = find(g(:,l)==1);
        if (~isempty(points))
            betag = beta(:,l);
            Ytildeg  = Ytilde(points,:);
            Xg  = squeeze(X(points,:,:));
            
            sse = Ytildeg-Xg*betag;
            
            s_sigg = s_sig + 0.5*length(points);
            r_sigg = r_sig + 0.5*(sse'*sse);
            
            sigmav(1,l) = 1./gamrnd(s_sigg,1/r_sigg);
        end
    end
    
    sigmav_draws(iter,:) = sigmav;
    %  toc
    % =====| STEP 3
    % =====| Sample rho using M-H
    % tic;
    % disp(['Sampling rho...'])
    
    for l=randperm(ncomp)
        rho_p = rho_draws(iter-1,:);
        points = find(g(:,l)==1);
        if (~isempty(points))
            betag = beta(:,l);
            sigmavg = sigmav(1,l);
            % Propose new rho
            accept_r = 0;
            while (accept_r==0)
               if iter < 2
                   rho_p(1,l) = mvnrnd(rho_draws(iter-1,l), eye(1)*0.1^2);
               else
                   rv = rand(1);
                   rho_p(1,l) = mvnrnd(rho_draws(iter-1,l), cov(rho_draws(1:iter-1,l))*2.38^2)*(rv<=0.95)...
                       + mvnrnd(rho_draws(iter-1,l),eye(1)*0.1^2)*(rv>0.95);
               end
               if(abs( rho_p(1,l) )<1)
                   accept_r =1;
               end
            end
            rho_p_draws(iter,:) = rho_p;
            pp_r = 0;
            % Ratio of the likelihood function
            rho0_til = g*rho_draws(iter-1,:)';
            rho1_til = g*rho_p';
            pp_r = logliknc_ratio_mixSAR(Y, X, Wr, rho0_til, rho1_til, betag, sigmavg, points);
            % Ratio of priors for rho
            pp_r = pp_r + betacenlpdf(rho_p(1,l),d_rho) - betacenlpdf(rho_draws(iter-1,l),d_rho);
            
            % Transition to candidate rho1(l) with probability
            pp_r = min(pp_r, 0);
            
            if (log(rand(1))<pp_r)
                rho_draws(iter,l) = rho_p(1,l);
                acc_rho(1,l) = acc_rho(1,l) + 1;
            else
                rho_draws(iter,l) = rho_draws(iter-1,l);
            end
            
            accrate_rho(iter,l) = acc_rho(1,l)/iter;
            
        end
    end %l
    
    rho = rho_draws(iter,:);
    rho_til = g*rho';
    % toc
    
    % =====| STEP 4
    % =====| Sample component label vectors
    % tic;
    % disp(['Sampling component label...'])
    dens = zeros(N,ncomp);
    prob = zeros(N,ncomp);
    
    for l=1:ncomp
        betag = beta(:,l);
        sigmavg = sigmav(1,l);
        rhog = rho(1,l);
        omegag = omega(:,l);
        
        for i=1:N
            dens(i,l) = omegag*normpdf(Y(i,:)-rhog*Wr(i,:)*Y-X(i,:)*betag,0,sqrt(sigmavg));
        end
    end
    
    for l = 1:ncomp
        prob(:,l) = dens(:,l)./sum(dens,2);
    end
    prob_draws(iter,:,:) = prob;
    g = mnrnd(1,prob);
    g_draws(iter,:,:) = g;
    %  toc
    
end %iter

%% Save results
beta_save   = beta_draws(nburn+1:nsave+nburn,:,:);
beta1_save  = squeeze(beta_draws(nburn+1:nsave+nburn,1,:));
beta2_save  = squeeze(beta_draws(nburn+1:nsave+nburn,2:end,:));
sigmav_save = sigmav_draws(nburn+1:nsave+nburn,:);
rho_save    = rho_draws(nburn+1:nsave+nburn,:);
g_save      = g_draws(nburn+1:nsave+nburn,:,:);
omega_save  = omega_draws(nburn+1:nsave+nburn,:);

beta1_mean_mix  = reshape(mean(beta1_save,1),1,ncomp);
beta2_mean_mix  = reshape(mean(beta2_save,1),k-1,ncomp);
sigmav_mean_mix = reshape(mean(sigmav_save,1),1,ncomp);
rho_mean_mix    = reshape(mean(rho_save,1),1,ncomp);
g_mean_mix      = reshape(mean(g_save,1),N,ncomp);
omega_mean_mix  = mean(omega_save,1)';


%% Relabeling
sigmav_save_temp = reshape(sigmav_save,nsave,1,3);
res_save = [beta_save];
% res_save = [beta_save,sigmav_save_temp];

parclust = res_save;
parclust=permute(parclust,[3 1 2]); %permute such that first index is equal to group, second index to MCMC iteratio, third index to the components of group specific parameter
parclust=reshape(parclust,size(parclust,1)*size(parclust,2),size(parclust,3));
[S,clu]=kmeans(parclust,ncomp,'EmptyAction','singleton');
perm=reshape(S,ncomp,size(S,1)/ncomp)'; % reshape nsave times ncomp

clear parperm
parperm.beta1_save = beta1_save;
parperm.beta2_save = beta2_save;
parperm.rho_save   = rho_save;
parperm.sigmav_save = sigmav_save;
parperm.omega_save = omega_save;
for l = 1:ncomp
   il = (perm==l*ones(nsave,ncomp));
   parperm.beta1_save(:,l) = sum(il.*beta1_save,2);
   parperm.beta2_save(:,l) = sum(il.*beta2_save,2);
   parperm.rho_save (:,l)  = sum(il.*rho_save,2);
   parperm.sigmav_save(:,l) = sum(il.*sigmav_save,2);
   parperm.omega_save(:,l) = sum(il.*omega_save,2);
end

parperm.omega_mean_mix = mean(parperm.omega_save,1);
[parperm.omega_mean_mix,ind] = sort(parperm.omega_mean_mix,'descend');
parperm.omega_save = parperm.omega_save(:,ind);
parperm.beta1_save = parperm.beta1_save(:,ind);
parperm.beta2_save = parperm.beta2_save(:,ind);
parperm.rho_save = parperm.rho_save(:,ind);
parperm.sigmav_save = parperm.sigmav_save(:,ind);

parperm.beta1_mean_mix = mean(parperm.beta1_save,1);
parperm.beta2_mean_mix = mean(parperm.beta2_save,1);
parperm.rho_mean_mix = mean(parperm.rho_save,1);
parperm.sigmav_mean_mix = mean(parperm.sigmav_save,1);

%% Trace plots of coefficients
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

% rho
figure
subplot(3,1,1)
plot(1:nsave,rho_save(:,1)); 
hold on; yline(rho_true(1),'Linewidth',2,'Color',[.6 0 0]); 
hold on; yline(rho_mean_mix(1),'Linewidth',2);
ylabel('\rho_{1}, g=1');xlabel(['iter; true = ', num2str(rho_true(1)), ', est = ', num2str(rho_mean_mix(1))])
subplot(3,1,2)
plot(1:nsave,rho_save(:,2)); 
hold on; yline(rho_true(2),'Linewidth',2,'Color',[.6 0 0]); 
hold on; yline(rho_mean_mix(2),'Linewidth',2);
ylabel('\rho_{2}, g=2');xlabel(['iter; true = ', num2str(rho_true(2)), ', est = ', num2str(rho_mean_mix(2))])
subplot(3,1,3)
plot(1:nsave,rho_save(:,3)); 
hold on; yline(rho_true(3),'Linewidth',2,'Color',[.6 0 0]); 
hold on; yline(rho_mean_mix(3),'Linewidth',2);
ylabel('\rho_{3}, g=3');xlabel(['iter; true = ', num2str(rho_true(3)), ', est = ', num2str(rho_mean_mix(3))])

figure
plot(1:nsave,rho_save(:,1)); 
hold on;
plot(1:nsave,rho_save(:,2)); 
hold on;
plot(1:nsave,rho_save(:,3)); 
ylabel('\rho');xlabel('iter');
legend('\rho_{1}','\rho_{2}','\rho_{3}');

% sigmasq
figure
subplot(3,1,1)
plot(1:nsave,sigmav_save(:,1)); hold on; yline(sigmasq_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmav_mean_mix(1),'Linewidth',2);
ylabel('\sigma^2_{1}, g=1');xlabel(['iter; true = ', num2str(sigmasq_true(1)), ', est = ', num2str(sigmav_mean_mix(1))])
subplot(3,1,2)
plot(1:nsave,sigmav_save(:,2)); hold on; yline(sigmasq_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmav_mean_mix(2),'Linewidth',2);
ylabel('\sigma^2_{2}, g=2');xlabel(['iter; true = ', num2str(sigmasq_true(2)), ', est = ', num2str(sigmav_mean_mix(2))])
subplot(3,1,3)
plot(1:nsave,sigmav_save(:,3)); hold on; yline(sigmasq_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmav_mean_mix(3),'Linewidth',2);
ylabel('\sigma^2_{3}, g=3');xlabel(['iter; true = ', num2str(sigmasq_true(3)), ', est = ', num2str(sigmav_mean_mix(3))])

figure
plot(1:nsave,sigmav_save(:,1)); 
hold on;
plot(sigmav_save(:,2)); 
hold on;
plot(sigmav_save(:,3)); 
ylabel('\sigma^2');xlabel('iter');
legend('\sigma^2_{1}','\sigma^2_{2}','\sigma^2_{3}');

% omega
figure
subplot(3,1,1)
plot(1:nsave,omega_save(:,1)); hold on; yline(omega_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_mean(1),'Linewidth',2);
ylabel('\omega_{1}, g=1');xlabel(['iter; true = ', num2str(omega_true(1)), ', est = ', num2str(omega_mean(1))])
subplot(3,1,2)
plot(1:nsave,omega_save(:,2)); hold on; yline(omega_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_mean(2),'Linewidth',2);
ylabel('\omega_{2}, g=2');xlabel(['iter; true = ', num2str(omega_true(2)), ', est = ', num2str(omega_mean(2))])
subplot(3,1,3)
plot(1:nsave,omega_save(:,3)); hold on; yline(omega_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_mean(3),'Linewidth',2);
ylabel('\omega_{3}, g=3');xlabel(['iter; true = ', num2str(omega_true(3)), ', est = ', num2str(omega_mean(3))])                                                                         

figure
plot(1:nsave,omega_save(:,1)); 
hold on;
plot(1:nsave,omega_save(:,2)); 
hold on;
plot(1:nsave,omega_save(:,3)); 
ylabel('\omega');xlabel('iter');
legend('\omega_{1}','\omega_{2}','\omega_{3}');


% M-H acceptance rate
figure
plot(1:nsave+nburn,accrate_rho(:,1)); 
hold on; 
plot(1:nsave+nburn,accrate_rho(:,2)); 
hold on; 
plot(1:nsave+nburn,accrate_rho(:,3)); 
legend('Group 1 Acceptance Rate', 'Group 2 Acceptance Rate', 'Group 3 Acceptance Rate')
legend('Location', 'northoutside')


%% Trace plots of coefficients after permutation
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

% parperm.rho
figure
subplot(3,1,1)
plot(1:nsave,parperm.rho_save(:,1)); hold on; yline(rho_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.rho_mean_mix(1),'Linewidth',2);
ylabel('\rho_{1}, g=1');xlabel(['iter; true = ', num2str(rho_true(1)), ', est = ', num2str(parperm.rho_mean_mix(1))]);xlim([0 nsave]);
subplot(3,1,2)
plot(1:nsave,parperm.rho_save(:,2)); hold on; yline(rho_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.rho_mean_mix(2),'Linewidth',2);
ylabel('\rho_{2}, g=2');xlabel(['iter; true = ', num2str(rho_true(2)), ', est = ', num2str(parperm.rho_mean_mix(2))]);xlim([0 nsave]);
subplot(3,1,3)
plot(1:nsave,parperm.rho_save(:,3)); hold on; yline(rho_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.rho_mean_mix(3),'Linewidth',2);
ylabel('\rho_{3}, g=3');xlabel(['iter; true = ', num2str(rho_true(3)), ', est = ', num2str(parperm.rho_mean_mix(3))]);xlim([0 nsave]);

F1{2,2} = figure('visible','off');
plot(1:nsave,parperm.rho_save(:,1)); 
hold on;
plot(1:nsave,parperm.rho_save(:,2)); 
hold on;
plot(1:nsave,parperm.rho_save(:,3)); 
ylabel('\rho');xlabel('iter');
legend('\rho_{1}','\rho_{2}','\rho_{3}');

% sigmav
figure
subplot(3,1,1)
plot(1:nsave,parperm.sigmav_save(:,1)); hold on; yline(sigmasq_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmav_mean_mix(1),'Linewidth',2);
ylabel('\sigma^2_v_{1}, g=1');xlabel(['iter; true = ', num2str(sigmasq_true(1)), ', est = ', num2str(parperm.sigmav_mean_mix(1))]);xlim([0 nsave]);
subplot(3,1,2)
plot(1:nsave,parperm.sigmav_save(:,2)); hold on; yline(sigmasq_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmav_mean_mix(2),'Linewidth',2);
ylabel('\sigma^2_v_{2}, g=2');xlabel(['iter; true = ', num2str(sigmasq_true(2)), ', est = ', num2str(parperm.sigmav_mean_mix(2))]);xlim([0 nsave]);
subplot(3,1,3)
plot(1:nsave,parperm.sigmav_save(:,3)); hold on; yline(sigmasq_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmav_mean_mix(3),'Linewidth',2);
ylabel('\sigma^2_v_{3}, g=3');xlabel(['iter; true = ', num2str(sigmasq_true(3)), ', est = ', num2str(parperm.sigmav_mean_mix(3))]);xlim([0 nsave]);

figure
plot(1:nsave,parperm.sigmav_save(:,1)); 
hold on;
plot(1:nsave,parperm.sigmav_save(:,2)); 
hold on;
plot(1:nsave,parperm.sigmav_save(:,3)); 
ylabel('\sigma^2_v');xlabel('iter');
legend('\sigma^2_v_{1}','\sigma^2_v_{2}','\sigma^2_v_{3}')

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



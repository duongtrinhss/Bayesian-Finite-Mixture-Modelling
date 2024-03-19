% =====================================================================================
% Written by Duong Trinh
% University of Glasgow
% This version: June 2023
% =====================================================================================
clear all; clc; close all;
ssd = 20232907;
% rng(ssd)
addpath(genpath('functions'))

%% DGP
options.form = 1; %mixture
options.heter = 3;
options.R2 = 0.75; 
n = 2000;

[Y_gen,X_gen,g_true,beta_true,sigmasq_true,omega_true,extout] = genDataMLR(n,options);

y    = Y_gen;
xmat = X_gen;

%% ESTIMATIONS
nsave = 10000;
nburn = 1000;
% =====| Method for permutation
perm_method = 'kmeanclust';
% 'ecr'
% 'aic'
% 'ecrIterative1'
% 'kmeanclust'
% 'dataBased'
opt = 1;

[resFullmcmc,resMean,resMedian] = mcmcBayesianMixtureMLR(y,xmat,nsave,nburn);
[parperm] = permutemcmcBayesianMixtureMLR(y,resFullmcmc,perm_method,opt);

%% PROTOTYPE
%==========================================================================
%============================| PRELIMINARIES |=============================
nsave = 5000;
nburn = 500;
n     = size(y,1);
k     = size(xmat,2);

% =====| Initialize parameters
ncomp  = 3;
omega  = (1/ncomp)*ones(1,ncomp);
g      = mnrnd(1,omega,n);
alpha  = ones(1,ncomp);

% coefficients
mu_b = zeros(k,1); Q_b = 10*ones(k,1); 
beta = .5*ones(k,ncomp);

% covariances
sigmasq = ones(1,ncomp); s_sig = 3; r_sig = 2;

% =====| Storage space for MCMC

% coefficients
beta_draws    = zeros(nsave+nburn,k,ncomp);

% variances
sigmasq_draws = zeros(nsave+nburn,1,ncomp);

% probability
omega_draws   = zeros(nsave+nburn,1,ncomp);

% component labels
g_draws       = zeros(nsave+nburn,n,ncomp);
prob_draws    = zeros(nsave+nburn,n,ncomp);


%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================
tic;
for iter = 1:nsave+nburn
    
    if mod(iter,100)==0   
        disp(['This is iteration ' num2str(iter)])         
        disp([num2str(100*(iter/(nsave+nburn))) '% completed'])
        toc
    end   
    
    % =====| Draw compnent probability
    nn                  = sum(g,1);
    omega               = drchrnd(nn + alpha,1);
%     [omega, ind]        = sort(omega, 'descend');
%     g                   = g(:,ind);
%     omega = omega_true';
    
    omega_draws(iter,:) = omega;
    
    % =====| Draw the coefficient vectors
    % [beta]
    
    for l = 1:ncomp
        points   = find(g(:,l)==1);
        if (~isempty(points)==1)
            xg       = xmat(points,:);
            yg       = y(points,:);
            sigmasqg = sigmasq(l);
            
            V_bg  = ( diag(1./Q_b) + xg'*xg/sigmasqg )\eye(k);
            mu_bg = V_bg*( diag(1./Q_b)*mu_b + xg'*yg/sigmasqg );
            beta(:,l) = mu_bg + chol(V_bg)'*randn(k,1);       
        end
    end
    
    beta_draws(iter,:,:) = beta;
   
    % =====| Draw variance 
    % [sigmasq]
    
    for l=1:ncomp
        points = find(g(:,l)==1);
         if (~isempty(points)==1)
            xg       = xmat(points,:);
            yg       = y(points,:);
            betag = beta(:,l);
            
            sse = yg-xg*betag;
            
            s_sigg = s_sig + 0.5*length(points);
            r_sigg = r_sig + 0.5*(sse'*sse);
            
            sigmasq(1,l) = 1./gamrnd(s_sigg,1/r_sigg);
        end       
    end

    sigmasq_draws(iter,:,:) = sigmasq;
    
    % =====| Draw component latent vectors
%     dens = zeros(n,ncomp);
    kers = zeros(n,ncomp);
    for l=1:ncomp
        betag = beta(:,l);
        sigmasqg = sigmasq(l);
        omegag = omega(:,l);
            
%         dens(:,l) = normpdf(y-xmat*betag,0,sqrt(sigmasqg));
%         for i=1:n
            kers(:,l) = log(omegag) + lnormpdf(y-xmat*betag,0,sqrt(sigmasqg));
%         end
    clear betag sigmasqg omegag
    end 
%     prob              = omega.*dens;
%     prob              = prob./sum(prob,2);
    
    kertrans = exp(kers - max(kers,[],2));
    prob = kertrans./sum(kertrans,2);

    g                 = mnrnd(1,prob);
    
%     g = g_true;
    prob_draws(iter,:,:) = prob;
    g_draws(iter,:,:) = g;
    
%     rnd = rand(N,1);
%     g = sum(cumsum(prob,2) < rnd(:,ones(K,1)),2) + 1;

%     g = g_gen;

%     iperm   = randperm(ncomp);
%     omega   = omega(iperm);
%     beta    = beta(:,iperm);
%     sigmasq = sigmasq(iperm);
%     g       = g(:,iperm);
%     
%     omega_draws(iter,:)     = omega;
%     beta_draws(iter,:,:)    = beta;
%     sigmasq_draws(iter,:)   = sigmasq;
%     g_draws(iter,:,:)       = g;
    
end

%% Save results
% =====| Estimation
beta_save = beta_draws(nburn+1:nsave+nburn,:,:);
beta1_save = squeeze(beta_draws(nburn+1:nsave+nburn,1,:));
beta1_mean_mix = reshape(mean(beta1_save,1),1,ncomp);
beta2_save = squeeze(beta_draws(nburn+1:nsave+nburn,2,:));
beta2_mean_mix = reshape(mean(beta2_save,1),1,ncomp);

sigmasq_save = sigmasq_draws(nburn+1:nburn+nsave,:);
sigmasq_mean_mix = reshape(mean(sigmasq_save,1),1,ncomp);

omega_save = omega_draws(nburn+1:nsave+nburn,:);
omega_mean = mean(omega_save,1)';

g_save = g_draws(nburn+1:nsave+nburn,:,:);
g_mean = squeeze(mean(g_save,1));

prob_save = prob_draws(nburn+1:nsave+nburn,:,:);

% =====| Relabeling
sigmasq_save_temp = reshape(sigmasq_save,nsave,1,3);

res_save = [beta_save];
% res_save = [beta_save,sigmasq_save_temp];

parclust = res_save;
parclust=permute(parclust,[3 1 2]); %permute such that first index is equal to group, second index to MCMC iteratio, third index to the components of group specific parameter
parclust=reshape(parclust,size(parclust,1)*size(parclust,2),size(parclust,3));
[S,clu]=kmeans(parclust,ncomp,'EmptyAction','singleton');
perm=reshape(S,ncomp,size(S,1)/ncomp)'; % reshape nsave times ncomp

clear parperm
parperm.beta1_save = beta1_save;
parperm.beta2_save = beta2_save;
parperm.sigmasq_save = sigmasq_save;
parperm.omega_save = omega_save;
for l = 1:ncomp
   il = (perm==l*ones(nsave,ncomp));
   parperm.beta1_save(:,l) = sum(il.*beta1_save,2);
   parperm.beta2_save(:,l) = sum(il.*beta2_save,2);
   parperm.sigmasq_save(:,l) = sum(il.*sigmasq_save,2);
   parperm.omega_save(:,l) = sum(il.*omega_save,2);
end

parperm.omega_mean_mix = mean(parperm.omega_save,1);
[parperm.omega_mean_mix,ind] = sort(parperm.omega_mean_mix,'descend');
parperm.omega_save = parperm.omega_save(:,ind);
parperm.beta1_save = parperm.beta1_save(:,ind);
parperm.beta2_save = parperm.beta2_save(:,ind);
parperm.sigmasq_save = parperm.sigmasq_save(:,ind);

parperm.beta1_mean_mix = mean(parperm.beta1_save,1);
parperm.beta2_mean_mix = mean(parperm.beta2_save,1);
parperm.sigmasq_mean_mix = mean(parperm.sigmasq_save,1);
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

% sigmasq
figure
subplot(3,1,1)
plot(1:nsave,sigmasq_save(:,1)); hold on; yline(sigmasq_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_mean_mix(1),'Linewidth',2);
ylabel('\sigma^2_{1}, g=1');xlabel(['iter; true = ', num2str(sigmasq_true(1)), ', est = ', num2str(sigmasq_mean_mix(1))])
subplot(3,1,2)
plot(1:nsave,sigmasq_save(:,2)); hold on; yline(sigmasq_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_mean_mix(2),'Linewidth',2);
ylabel('\sigma^2_{2}, g=2');xlabel(['iter; true = ', num2str(sigmasq_true(2)), ', est = ', num2str(sigmasq_mean_mix(2))])
subplot(3,1,3)
plot(1:nsave,sigmasq_save(:,3)); hold on; yline(sigmasq_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_mean_mix(3),'Linewidth',2);
ylabel('\sigma^2_{3}, g=3');xlabel(['iter; true = ', num2str(sigmasq_true(3)), ', est = ', num2str(sigmasq_mean_mix(3))])

figure
plot(1:nsave,sigmasq_save(:,1)); 
hold on;
plot(sigmasq_save(:,2)); 
hold on;
plot(sigmasq_save(:,3)); 
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
legend('\beta_{21}','\beta_{22}','\beta_{23}');

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



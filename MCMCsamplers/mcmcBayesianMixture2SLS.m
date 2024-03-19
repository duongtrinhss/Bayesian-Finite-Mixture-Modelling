function [resFull,resMean,resMedian] = mcmcBayesianMixture2SLS(Y,D,zmat,xmat,nsave,nburn,g_true)
% mcmcBayesianMixture2SLS
% INPUT:

if (~exist('nsave','var')||~exist('nburn','var'))
    nsave = 5000;
    nburn = 500;
end

n = size(Y,1);
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
prob_draws = zeros(nsave+nburn,n,ncomp);

% component probability
omega_draws = zeros(nsave+nburn,ncomp);


%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================

tic;
for iter = 1:nsave+nburn
    
    if mod(iter,100)==0   
        disp(['This is iteration ' num2str(iter)])         
        disp([num2str(100*(iter/(nsave+nburn))) '% completed'])
        toc
    end
    
    % =====| STEP 1   
    % =====| Draw compnent probability
    nn                  = sum(g,1);
    omega               = drchrnd(nn + alpha,1);

    omega_draws(iter,:) = omega;
    
    % =====| STEP 2
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

    gamma_draws(iter,:,:) = gamma;
    beta_draws(iter,:,:)  = beta;

    % =====| STEP 3
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

    sigv_draws(iter,:)  = sigv;
    sigu_draws(iter,:)  = sigu;
    siguv_draws(iter,:) = siguv;

    % =====| STEP 4
    % =====| Draw component label vectors
    % {g}
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
        clear gammag betag Sigg Siginvg omegag
    end
    
    kertrans = exp(kers - max(kers,[],2));
    prob = kertrans./sum(kertrans,2);

    g                 = mnrnd(1,prob);
%     g = g_true;
    
    prob_draws(iter,:,:) = prob;
    g_draws(iter,:,:) = g;
    
end %iter

%% Save results
gamma_save = gamma_draws(nburn+1:nburn+nsave,:,:);
beta_save = beta_draws(nburn+1:nsave+nburn,:,:);

gamma1_save = squeeze(gamma_draws(nburn+1:nsave+nburn,1,:));
gamma2_save = squeeze(gamma_draws(nburn+1:nsave+nburn,2,:));
beta1_save = squeeze(beta_draws(nburn+1:nsave+nburn,1,:));
beta2_save = squeeze(beta_draws(nburn+1:nsave+nburn,2,:));

sigv_save  = sigv_draws(nburn+1:nburn+nsave,:);
sigu_save  = sigu_draws(nburn+1:nburn+nsave,:);
siguv_save = siguv_draws(nburn+1:nburn+nsave,:);

sigmasq_v_save = sigv_save.^2;
sigmasq_u_save = sigu_save.^2;
cov_uv_save    = siguv_save;

omega_save   = omega_draws(nburn+1:nsave+nburn,:);
g_save       = g_draws(nburn+1:nsave+nburn,:,:);
prob_save    = prob_draws(nburn+1:nsave+nburn,:,:);

resFull.gamma_save = gamma_save;
resFull.beta_save = beta_save;
resFull.gamma1_save = gamma1_save;
resFull.gamma2_save = gamma2_save;
resFull.beta1_save = beta1_save;
resFull.beta2_save = beta2_save;
resFull.sigmasq_v_save = sigmasq_v_save;
resFull.sigmasq_u_save = sigmasq_u_save;
resFull.cov_uv_save = cov_uv_save;
resFull.omega_save = omega_save;
resFull.g_save = g_save;
resFull.prob_save = prob_save;


% Sort and derive posterior mean
resMean.omega_mean_mix = mean(omega_save,1);
[resMean.omega_mean_mix,ind1] = sort(resMean.omega_mean_mix,2,'descend');
resMean.omega_save = omega_save(:,ind1);
resMean.gamma1_save = gamma1_save(:,ind1);
resMean.gamma2_save = gamma2_save(:,ind1);
resMean.beta1_save = beta1_save(:,ind1);
resMean.beta2_save = beta2_save(:,ind1);
resMean.sigmasq_v_save = sigmasq_v_save(:,ind1);
resMean.sigmasq_u_save = sigmasq_u_save(:,ind1);
resMean.cov_uv_save = cov_uv_save(:,ind1);
resMean.gamma1_mean_mix = reshape(mean(resMean.gamma1_save,1),1,ncomp);
resMean.gamma2_mean_mix = reshape(mean(resMean.gamma2_save,1),1,ncomp);
resMean.beta1_mean_mix = reshape(mean(resMean.beta1_save,1),1,ncomp);
resMean.beta2_mean_mix = reshape(mean(resMean.beta2_save,1),1,ncomp);
resMean.sigmasq_v_mean_mix  = mean(resMean.sigmasq_v_save,1);
resMean.sigmasq_u_mean_mix  = mean(resMean.sigmasq_u_save,1);
resMean.cov_uv_mean_mix     = mean(resMean.cov_uv_save,1);

% Sort and derive posterior median
resMedian.omega_median_mix = median(omega_save,1);
[resMedian.omega_median_mix,ind2] = sort(resMedian.omega_median_mix,2,'descend');
resMedian.omega_save = omega_save(:,ind2);
resMedian.gamma1_save = gamma1_save(:,ind2);
resMedian.gamma2_save = gamma2_save(:,ind2);
resMedian.beta1_save = beta1_save(:,ind2);
resMedian.beta2_save = beta2_save(:,ind2);
resMedian.sigmasq_v_save = sigmasq_v_save(:,ind2);
resMedian.sigmasq_u_save = sigmasq_u_save(:,ind2);
resMedian.cov_uv_save = cov_uv_save(:,ind2);
resMedian.gamma1_median_mix = reshape(median(resMedian.gamma1_save,1),1,ncomp);
resMedian.gamma2_median_mix = reshape(median(resMedian.gamma2_save,1),1,ncomp);
resMedian.beta1_median_mix = reshape(median(resMedian.beta1_save,1),1,ncomp);
resMedian.beta2_median_mix = reshape(median(resMedian.beta2_save,1),1,ncomp);
resMedian.sigmasq_v_median_mix  = median(resMedian.sigmasq_v_save,1);
resMedian.sigmasq_u_median_mix  = median(resMedian.sigmasq_u_save,1);
resMedian.cov_uv_median_mix     = median(resMedian.cov_uv_save,1);
end
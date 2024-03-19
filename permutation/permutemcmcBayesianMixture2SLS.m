function [perm,parperm] = permutemcmcBayesianMixture2SLS(Y,resFullmcmc,est_model,perm_method,opt)
% permutemcmcBayesianMixtureMLR: permutations after MCMC
% INPUT:
% perm_method - Method for label-switching issue           
% opt         - 1 if Median, 2 if Mean

% Preliminary
gamma_save    = resFullmcmc.gamma_save;
beta_save     = resFullmcmc.beta_save;
if strcmp(est_model,'BayesianMixture2SLS')
    gamma1_save   = resFullmcmc.gamma1_save;
    gamma2_save   = resFullmcmc.gamma2_save;
    sigmasq_v_save = resFullmcmc.sigmasq_v_save;
    sigmasq_u_save = resFullmcmc.sigmasq_u_save;
    cov_uv_save  = resFullmcmc.cov_uv_save;
elseif strcmp(est_model,'BayesianMixture2SLSSSN')
    sigmasq_v_save = resFullmcmc.sigvsq_save;
    sigmasq_u_save = resFullmcmc.sigusq_save;
    cov_uv_save = resFullmcmc.siguv_save;
end
beta1_save    = resFullmcmc.beta1_save;
beta2_save    = resFullmcmc.beta2_save;
g_save       = resFullmcmc.g_save;
prob_save    = resFullmcmc.prob_save;
omega_save   = resFullmcmc.omega_save;

nsave = size(omega_save,1);
ncomp = size(omega_save,2); %ncomp
K = ncomp;

if (strcmp(perm_method,'ecr')||strcmp(perm_method,'ecrIterative1')||strcmp(perm_method,'ecrIterative2')||strcmp(perm_method,'databased')) 
    
    g_MAP  = squeeze(median(g_save,1));
    zpivot = g_MAP*[1 2 3]';
    
    zmcmc = zeros(size(g_save,1,2));
    for l=1:ncomp
        zmcmc = zmcmc + g_save(:,:,l)*l;
    end
    
    p = prob_save;
    
    threshold = 1e-6;
    maxiter = 100;
end

% Permutations

switch perm_method
    case 'aic'
        param = omega_save;
        perm = aic(param,K);
    case 'ecr'
        [~,perm] = ecr(zpivot,zmcmc,K);
    case 'ecrIterative1'
        [~,perm] = ecrIterative1(zmcmc,K,threshold,maxiter);
    case 'ecrIterative2'
        [~,perm] = ecrIterative2(zmcmc,K,p,threshold,maxiter);
    case 'kmeanclust'
        param = [omega_save];
        parclust = permute(param,[1,3,2]);
        [perm] = kmeanclust(parclust,K);
    case 'databased'
        [perm] = dataBased(Y,K,zmcmc);
end  

% Relabelling

clear parperm

parperm.beta1_save = beta1_save;
parperm.beta2_save = beta2_save;
if strcmp(est_model,'BayesianMixture2SLS')
    parperm.gamma1_save = gamma1_save;
    parperm.gamma2_save = gamma2_save;
end
parperm.omega_save = omega_save;
parperm.sigmasq_v_save = sigmasq_v_save;
parperm.sigmasq_u_save = sigmasq_u_save;
parperm.cov_uv_save = cov_uv_save;

for l = 1:ncomp
   il = (perm==l*ones(nsave,ncomp));
   parperm.beta1_save(:,l) = sum(il.*beta1_save,2);
   parperm.beta2_save(:,l) = sum(il.*beta2_save,2);
if strcmp(est_model,'BayesianMixture2SLS')
   parperm.gamma1_save(:,l) = sum(il.*gamma1_save,2);
   parperm.gamma2_save(:,l) = sum(il.*gamma2_save,2);
end
   parperm.omega_save(:,l) = sum(il.*omega_save,2);
   parperm.sigmasq_v_save(:,l) = sum(il.*sigmasq_v_save,2);
   parperm.sigmasq_u_save(:,l) = sum(il.*sigmasq_u_save,2);
   parperm.cov_uv_save(:,l) = sum(il.*cov_uv_save,2);
end

if(opt==1)
    % Median
    parperm.omega_median_mix = median(parperm.omega_save,1);
    [parperm.omega_median_mix,ind] = sort(parperm.omega_median_mix,'descend');
    parperm.omega_save = parperm.omega_save(:,ind);
    
    parperm.beta1_save = parperm.beta1_save(:,ind);
    parperm.beta2_save = parperm.beta2_save(:,ind);
if strcmp(est_model,'BayesianMixture2SLS')
    parperm.gamma1_save = parperm.gamma1_save(:,ind);
    parperm.gamma2_save = parperm.gamma2_save(:,ind);
end
    parperm.sigmasq_v_save = parperm.sigmasq_v_save(:,ind);
    parperm.sigmasq_u_save = parperm.sigmasq_u_save(:,ind);
    parperm.cov_uv_save = parperm.cov_uv_save(:,ind);
    
    parperm.beta1_median_mix = median(parperm.beta1_save,1);
    parperm.beta2_median_mix = median(parperm.beta2_save,1);
if strcmp(est_model,'BayesianMixture2SLS')
    parperm.gamma1_median_mix = median(parperm.gamma1_save,1);
    parperm.gamma2_median_mix = median(parperm.gamma2_save,1);
end
    parperm.sigmasq_v_median_mix = median(parperm.sigmasq_v_save,1);
    parperm.sigmasq_u_median_mix = median(parperm.sigmasq_u_save,1);
    parperm.cov_uv_median_mix = median(parperm.cov_uv_save,1);
elseif(opt==2)
    % Mean
    parperm.omega_mean_mix = mean(parperm.omega_save,1);
    [parperm.omega_mean_mix,ind] = sort(parperm.omega_mean_mix,'descend');
    parperm.omega_save = parperm.omega_save(:,ind);
    
    parperm.beta1_save = parperm.beta1_save(:,ind);
    parperm.beta2_save = parperm.beta2_save(:,ind);
if strcmp(est_model,'BayesianMixture2SLS')
    parperm.gamma1_save = parperm.gamma1_save(:,ind);
    parperm.gamma2_save = parperm.gamma2_save(:,ind);
end
    parperm.sigmasq_v_save = parperm.sigmasq_v_save(:,ind);
    parperm.sigmasq_u_save = parperm.sigmasq_u_save(:,ind);
    parperm.cov_uv_save = parperm.cov_uv_save(:,ind);
    
    parperm.beta1_mean_mix = mean(parperm.beta1_save,1);
    parperm.beta2_mean_mix = mean(parperm.beta2_save,1);
if strcmp(est_model,'BayesianMixture2SLS')
    parperm.gamma1_mean_mix = mean(parperm.gamma1_save,1);
    parperm.gamma2_mean_mix = mean(parperm.gamma2_save,1);
end
    parperm.sigmasq_v_mean_mix = mean(parperm.sigmasq_v_save,1);
    parperm.sigmasq_u_mean_mix = mean(parperm.sigmasq_u_save,1);
    parperm.cov_uv_mean_mix = mean(parperm.cov_uv_save,1);
end

end
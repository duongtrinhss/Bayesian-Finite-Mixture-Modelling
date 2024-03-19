function [parperm] = permutemcmcBayesianMixtureMLR(y,resFullmcmc,perm_method,opt)
% permutemcmcBayesianMixtureMLR: permutations after MCMC
% INPUT:
% perm_method - Method for label-switching issue           
% opt         - 1 if Median, 2 if Mean

% Preliminary
beta_save    = resFullmcmc.beta_save;
beta1_save   = resFullmcmc.beta1_save;
beta2_save   = resFullmcmc.beta2_save;
sigmasq_save = resFullmcmc.sigmasq_save;
omega_save   = resFullmcmc.omega_save;
g_save       = resFullmcmc.g_save;
prob_save    = resFullmcmc.prob_save;

nsave = size(omega_save,1);
ncomp = size(omega_save,2); %ncomp
K = ncomp;

if (strcmp(perm_method,'ecr')||strcmp(perm_method,'ecrIterative1')||strcmp(perm_method,'ecrIterative2')||strcmp(perm_method,'dataBased'))
    
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
        param = [beta2_save];
        perm = aic(param,K);
    case 'ecr'
        [~,perm] = ecr(zpivot,zmcmc,K);
    case 'ecrIterative1'
        [~,perm] = ecrIterative1(zmcmc,K,threshold,maxiter);
    case 'ecrIterative2'
        [~,perm] = ecrIterative2(zmcmc,K,p,threshold,maxiter);
    case 'kmeanclust'
        parclust = [beta_save];
        perm = kmeanclust(parclust,K);
    case 'dataBased'
        [perm] = dataBased(y,K,zmcmc);
end  

% Relabelling

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

if(opt==1)
    % Median
    parperm.omega_median_mix = median(parperm.omega_save,1);
    [parperm.omega_median_mix,ind] = sort(parperm.omega_median_mix,'descend');
    parperm.omega_save = parperm.omega_save(:,ind);
    parperm.beta1_save = parperm.beta1_save(:,ind);
    parperm.beta2_save = parperm.beta2_save(:,ind);
    parperm.sigmasq_save = parperm.sigmasq_save(:,ind);
    
    parperm.beta1_median_mix = median(parperm.beta1_save,1);
    parperm.beta2_median_mix = median(parperm.beta2_save,1);
    parperm.sigmasq_median_mix = median(parperm.sigmasq_save,1);
elseif(opt==2)
    % Mean
    parperm.omega_mean_mix = mean(parperm.omega_save,1);
    [parperm.omega_mean_mix,ind] = sort(parperm.omega_mean_mix,'descend');
    parperm.omega_save = parperm.omega_save(:,ind);
    parperm.beta1_save = parperm.beta1_save(:,ind);
    parperm.beta2_save = parperm.beta2_save(:,ind);
    parperm.sigmasq_save = parperm.sigmasq_save(:,ind);
    
    parperm.beta1_mean_mix = mean(parperm.beta1_save,1);
    parperm.beta2_mean_mix = mean(parperm.beta2_save,1);
    parperm.sigmasq_mean_mix = mean(parperm.sigmasq_save,1);
end

end
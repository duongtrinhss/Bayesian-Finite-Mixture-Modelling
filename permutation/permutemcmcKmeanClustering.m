function [perm,parperm] = permutemcmcKmeanClustering(resFullmcmc,est_model,opt)
% permutemcmcKmeanClustering for BayesianMixture2SLS
% drop non-permutation

% PRELIMINARY
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

% RELABELLING
clear parperm

nvars = 1;
param = reshape([omega_save']',[nsave,nvars,ncomp]);
% parclust = permute(param,[1,3,2]);
parclust = param;
parclust = permute(parclust,[3 1 2]); %permute such that first index is equal to group, second index to MCMC iteratio, third index to the components of group specific parameter
parclust = reshape(parclust,size(parclust,1)*size(parclust,2),size(parclust,3));
parclust = zscore(parclust,1);
[S,clu]  = kmeans(parclust,ncomp,'EmptyAction','singleton');

idclass = reshape(S,ncomp,size(S,1)/ncomp)'; % reshape nsave times ncomp

isperm  = all(sort(idclass,2) == repmat([1:ncomp],length(idclass),1),2);
nonperm = sum(~isperm);

perm = idclass(isperm,:);

parperm.beta1_save = beta1_save(isperm,:);
parperm.beta2_save = beta2_save(isperm,:);
if strcmp(est_model,'BayesianMixture2SLS')
    parperm.gamma1_save = gamma1_save(isperm,:);
    parperm.gamma2_save = gamma2_save(isperm,:);
end
parperm.omega_save = omega_save(isperm,:);
parperm.sigmasq_v_save = sigmasq_v_save(isperm,:);
parperm.sigmasq_u_save = sigmasq_u_save(isperm,:);
parperm.cov_uv_save = cov_uv_save(isperm,:);

for l = 1:ncomp
   il = (perm==l*ones(sum(isperm),ncomp));
   parperm.beta1_save(:,l) = sum(il.*beta1_save(isperm,:),2);
   parperm.beta2_save(:,l) = sum(il.*beta2_save(isperm,:),2);
if strcmp(est_model,'BayesianMixture2SLS')
   parperm.gamma1_save(:,l) = sum(il.*gamma1_save(isperm,:),2);
   parperm.gamma2_save(:,l) = sum(il.*gamma2_save(isperm,:),2);
end
   parperm.omega_save(:,l) = sum(il.*omega_save(isperm,:),2);
   parperm.sigmasq_v_save(:,l) = sum(il.*sigmasq_v_save(isperm,:),2);
   parperm.sigmasq_u_save(:,l) = sum(il.*sigmasq_u_save(isperm,:),2);
   parperm.cov_uv_save(:,l) = sum(il.*cov_uv_save(isperm,:),2);
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

% fill non-perm by NaN for plotting
parperm.beta1_save = [parperm.beta1_save',repmat([NaN],nonperm,ncomp)']';
parperm.beta2_save = [parperm.beta2_save',repmat([NaN],nonperm,ncomp)']';
parperm.gamma1_save = [parperm.gamma1_save',repmat([NaN],nonperm,ncomp)']';
parperm.gamma2_save = [parperm.gamma2_save',repmat([NaN],nonperm,ncomp)']';
parperm.omega_save = [parperm.omega_save',repmat([NaN],nonperm,ncomp)']';
parperm.sigmasq_v_save = [parperm.sigmasq_v_save',repmat([NaN],nonperm,ncomp)']';
parperm.sigmasq_u_save = [parperm.sigmasq_u_save',repmat([NaN],nonperm,ncomp)']';
parperm.cov_uv_save = [parperm.cov_uv_save',repmat([NaN],nonperm,ncomp)']';

end
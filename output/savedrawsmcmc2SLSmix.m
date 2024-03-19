% Save MCMC results for FiniteMixture_2SLS setting

%% Create paths
% date = string(datetime("today"));
% date = strrep(date,"-","");
date = '31Jul2023';
result_path = ['Results_FiniteMixture_2SLS' '_' date];
mkdir(result_path);

spec = ['DGP',num2str(options.form),'/','N',num2str(n),'_','Heter',num2str(options.heter),'_','R2y',num2str(extout.R2_y*100)];

if (opt==1)
    method = [est_model,'_optmedian'];
elseif (opt==2)
    method = [est_model,'_optmean'];
end

result_subpath = [result_path,'/',spec,'/',method];
mkdir(result_subpath)

filename = [result_subpath,'/','estresult_',est_model];

%% Store results
genDataInfo.n = n;
genDataInfo.DGP    = options.form;
genDataInfo.R2     = extout.R2_y;
genDataInfo.Y_gen  = Y_gen;
genDataInfo.D_gen  = D_gen;
genDataInfo.z_gen  = z_gen;
genDataInfo.x_gen  = x_gen;
genDataInfo.sigmasq_v_true = sigmasq_v_true;
genDataInfo.sigmasq_u_true = sigmasq_u_true;
genDataInfo.cov_uv_true = cov_uv_true;
genDataInfo.sigmasq_e_true = sigmasq_e_true;
genDataInfo.g_true = extout.g;
genDataInfo.omega_true = extout.omega;

save(filename,'genDataInfo','resFullmcmc','resMean','resMedian')
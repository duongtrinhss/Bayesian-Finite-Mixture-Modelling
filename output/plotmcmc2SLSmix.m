% Generate Graphs: Convergence Diagnostics

%% Preliminary

%% Input
% date = string(datetime("today"));
% date = strrep(date,"-","");
date = '31Jul2023';
result_path = ['Results_FiniteMixture_2SLS' '_' date];
mkdir(result_path);

%% Setup style
n_coef = 3;
n_fig  = 2;
F1 = cell(n_coef,n_fig);

coef_names   = {'beta','gamma','omega','sigmasq_u','sigma_uv'};
fig_names    = {'trace','tracemerge'};


%% Load data, overall results, and estimator-specific results

spec = ['DGP',num2str(options.form),'/','N',num2str(n),'_','Heter',num2str(options.heter),'_','R2y',num2str(round(extout.R2_y,2)*100)];

if (opt==1)
    method = [est_model,'_optmedian'];
elseif (opt==2)
    method = [est_model,'_optmean'];
end

result_subpath = [result_path,'/',spec,'/',method];
mkdir(result_subpath)

if (opt==1)
    omega_save = resMedian.omega_save;
    beta1_save = resMedian.beta1_save;
    beta2_save = resMedian.beta2_save;
    if strcmp(est_model,'BayesianMixture2SLS')
        gamma1_save = resMedian.gamma1_save;
        gamma2_save = resMedian.gamma2_save;
        sigmasq_v_save = resMedian.sigmasq_v_save;
        sigmasq_u_save = resMedian.sigmasq_u_save;
        cov_uv_save = resMedian.cov_uv_save;
    elseif strcmp(est_model,'BayesianMixture2SLSSSN')
        gamma1_save  = resFullmcmc.gamma1_save;
        gamma2_save  = resFullmcmc.gamma2_save;
        sigmasq_v_save = resMedian.sigvsq_save;
        sigmasq_u_save = resMedian.sigusq_save;
        cov_uv_save = resMedian.siguv_save;
    elseif strcmp(est_model,'BayesianMixture2SLSnonlinear')
        gamma1_save  = resMedian.gamma1_save;
        gamma2_save  = resMedian.gamma2_save;
        sigmasq_v_save = resMedian.sigvsq_save;
        sigmasq_u_save = resMedian.sigusq_save;
        cov_uv_save = resMedian.siguv_save;
    end   
    omega_cen_mix = resMedian.omega_median_mix;
    beta1_cen_mix = resMedian.beta1_median_mix;
    beta2_cen_mix = resMedian.beta2_median_mix;
    if strcmp(est_model,'BayesianMixture2SLS')
        gamma1_cen_mix = resMedian.gamma1_median_mix;
        gamma2_cen_mix = resMedian.gamma2_median_mix;
        sigmasq_v_cen_mix = resMedian.sigmasq_v_median_mix;
        sigmasq_u_cen_mix = resMedian.sigmasq_u_median_mix;
        cov_uv_cen_mix = resMedian.cov_uv_median_mix;
    elseif strcmp(est_model,'BayesianMixture2SLSSSN')
        gamma1_cen  = resMedian.gamma1_median;
        gamma2_cen  = resMedian.gamma2_median;
        sigmasq_v_cen_mix = resMedian.sigvsq_median_mix;
        sigmasq_u_cen_mix = resMedian.sigusq_median_mix;
        cov_uv_cen_mix = resMedian.siguv_median_mix;
    elseif strcmp(est_model,'BayesianMixture2SLSnonlinear')
        gamma1_cen_mix = resMedian.gamma1_median_mix;
        gamma2_cen_mix = resMedian.gamma2_median_mix;
        sigmasq_v_cen_mix = resMedian.sigvsq_median_mix;
        sigmasq_u_cen_mix = resMedian.sigusq_median_mix;
        cov_uv_cen_mix = resMedian.siguv_median_mix;
    end
elseif (opt==2)
    omega_save = resMean.omega_save;
    beta1_save = resMean.beta1_save;
    beta2_save = resMean.beta2_save;
    if strcmp(est_model,'BayesianMixture2SLS')
        gamma1_save = resMean.gamma1_save;
        gamma2_save = resMean.gamma2_save;
        sigmasq_v_save = resMean.sigmasq_v_save;
        sigmasq_u_save = resMean.sigmasq_u_save;
        cov_uv_save = resMean.cov_uv_save;
    elseif strcmp(est_model,'BayesianMixture2SLSSSN')
        gamma1_save  = resFullmcmc.gamma1_save;
        gamma2_save  = resFullmcmc.gamma2_save;
        sigmasq_v_save = resMean.sigvsq_save;
        sigmasq_u_save = resMean.sigusq_save;
        cov_uv_save = resMean.siguv_save;
    elseif strcmp(est_model,'BayesianMixture2SLSnonlinear')
        gamma1_save = resMean.gamma1_save;
        gamma2_save = resMean.gamma2_save;
        sigmasq_v_save = resMean.sigvsq_save;
        sigmasq_u_save = resMean.sigusq_save;
        cov_uv_save = resMean.siguv_save;
    end    
    omega_cen_mix = resMean.omega_mean_mix;
    beta1_cen_mix = resMean.beta1_mean_mix;
    beta2_cen_mix = resMean.beta2_mean_mix;
    if strcmp(est_model,'BayesianMixture2SLS')
        gamma1_cen_mix = resMean.gamma1_mean_mix;
        gamma2_cen_mix = resMean.gamma2_mean_mix;
        sigmasq_v_cen_mix = resMean.sigmasq_v_mean_mix;
        sigmasq_u_cen_mix = resMean.sigmasq_u_mean_mix;
        cov_uv_cen_mix = resMean.cov_uv_mean_mix;
    elseif strcmp(est_model,'BayesianMixture2SLSSSN')
        gamma1_cen  = resMean.gamma1_mean;
        gamma2_cen  = resMean.gamma2_mean;
        sigmasq_v_cen_mix = resMedian.sigvsq_mean_mix;
        sigmasq_u_cen_mix = resMedian.sigusq_mean_mix;
        cov_uv_cen_mix = resMedian.siguv_mean_mix;
    elseif strcmp(est_model,'BayesianMixture2SLSnonlinear')
        gamma1_cen_mix = resMean.gamma1_mean_mix;
        gamma2_cen_mix = resMean.gamma2_mean_mix;
        sigmasq_v_cen_mix = resMedian.sigvsq_mean_mix;
        sigmasq_u_cen_mix = resMedian.sigusq_mean_mix;
        cov_uv_cen_mix = resMedian.siguv_mean_mix;
    end    
end

%% Plot graphs
% beta
F1{1,1} = figure('visible','off');
subplot(3,2,1)
plot(1:nsave,beta1_save(:,1)); hold on; yline(beta_true(1,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta1_cen_mix(1),'Linewidth',2);
ylabel('\beta_{11}, k=1, g=1');xlabel(['iter; true = ', num2str(beta_true(1,1)), ', est = ', num2str(beta1_cen_mix(1))]);xlim([0 nsave]);
subplot(3,2,3)
plot(1:nsave,beta1_save(:,2)); hold on; yline(beta_true(1,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta1_cen_mix(2),'Linewidth',2);
ylabel('\beta_{12}, k=1, g=2');xlabel(['iter; true = ', num2str(beta_true(1,2)), ', est = ', num2str(beta1_cen_mix(2))]);xlim([0 nsave]);
subplot(3,2,5)
plot(1:nsave,beta1_save(:,3)); hold on; yline(beta_true(1,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta1_cen_mix(3),'Linewidth',2);
ylabel('\beta_{13}, k=1, g=3');xlabel(['iter; true = ', num2str(beta_true(1,3)), ', est = ', num2str(beta1_cen_mix(3))]);xlim([0 nsave]);
subplot(3,2,2)
plot(1:nsave,beta2_save(:,1)); hold on; yline(beta_true(2,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta2_cen_mix(1),'Linewidth',2);
ylabel('\beta_{21}, k=2, g=1');xlabel(['iter; true = ', num2str(beta_true(2,1)), ', est = ', num2str(beta2_cen_mix(1))]);xlim([0 nsave]);
subplot(3,2,4)
plot(1:nsave,beta2_save(:,2)); hold on; yline(beta_true(2,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta2_cen_mix(2),'Linewidth',2);
ylabel('\beta_{22}, k=2, g=2');xlabel(['iter; true = ', num2str(beta_true(2,2)), ', est = ', num2str(beta2_cen_mix(2))]);xlim([0 nsave]);
subplot(3,2,6)
plot(1:nsave,beta2_save(:,3)); hold on; yline(beta_true(2,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(beta2_cen_mix(3),'Linewidth',2);
ylabel('\beta_{23}, k=2, g=3');xlabel(['iter; true = ', num2str(beta_true(2,3)), ', est = ', num2str(beta2_cen_mix(3))]);xlim([0 nsave]);

F1{1,2} = figure('visible','off');
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

% gamma
if (strcmp(est_model,'BayesianMixture2SLS')||strcmp(est_model,'BayesianMixture2SLSnonlinear'))
    F1{2,1} = figure('visible','off');
    subplot(3,2,1)
    plot(1:nsave,gamma1_save(:,1)); hold on; yline(gamma_true(1,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma1_cen_mix(1),'Linewidth',2);
    ylabel('\gamma_{11}, k=1, g=1');xlabel(['iter; true = ', num2str(gamma_true(1,1)), ', est = ', num2str(gamma1_cen_mix(1))]);xlim([0 nsave]);
    subplot(3,2,3)
    plot(1:nsave,gamma1_save(:,2)); hold on; yline(gamma_true(1,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma1_cen_mix(2),'Linewidth',2);
    ylabel('\gamma_{12}, k=1, g=2');xlabel(['iter; true = ', num2str(gamma_true(1,2)), ', est = ', num2str(gamma1_cen_mix(2))]);xlim([0 nsave]);
    subplot(3,2,5)
    plot(1:nsave,gamma1_save(:,3)); hold on; yline(gamma_true(1,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma1_cen_mix(3),'Linewidth',2);
    ylabel('\gamma_{13}, k=1, g=3');xlabel(['iter; true = ', num2str(gamma_true(1,3)), ', est = ', num2str(gamma1_cen_mix(3))]);xlim([0 nsave]);
    subplot(3,2,2)
    plot(1:nsave,gamma2_save(:,1)); hold on; yline(gamma_true(2,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma2_cen_mix(1),'Linewidth',2);
    ylabel('\gamma_{21}, k=2, g=1');xlabel(['iter; true = ', num2str(gamma_true(2,1)), ', est = ', num2str(gamma2_cen_mix(1))]);xlim([0 nsave]);
    subplot(3,2,4)
    plot(1:nsave,gamma2_save(:,2)); hold on; yline(gamma_true(2,2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma2_cen_mix(2),'Linewidth',2);
    ylabel('\gamma_{22}, k=2, g=2');xlabel(['iter; true = ', num2str(gamma_true(2,2)), ', est = ', num2str(gamma2_cen_mix(2))]);xlim([0 nsave]);
    subplot(3,2,6)
    plot(1:nsave,gamma2_save(:,3)); hold on; yline(gamma_true(2,3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma2_cen_mix(3),'Linewidth',2);
    ylabel('\gamma_{23}, k=2, g=3');xlabel(['iter; true = ', num2str(gamma_true(2,3)), ', est = ', num2str(gamma2_cen_mix(3))]);xlim([0 nsave]);
    
    F1{2,2} = figure('visible','off');
    plot(1:nsave,gamma1_save(:,1));
    hold on;
    plot(1:nsave,gamma1_save(:,2));
    hold on;
    plot(1:nsave,gamma1_save(:,3));
    hold on;
    plot(1:nsave,gamma2_save(:,1));
    hold on;
    plot(1:nsave,gamma2_save(:,2));
    hold on;
    plot(1:nsave,gamma2_save(:,3));
    hold on;
    ylabel('\gamma');xlabel('iter');
    legend('\gamma_{11}','\gamma_{12}','\gamma_{13}','\gamma_{21}','\gamma_{22}','\gamma_{23}');
    
elseif strcmp(est_model,'BayesianMixture2SLSSSN')
    F1{2,1} = figure('visible','off');
    plot(1:nsave,gamma1_save(:,1)); hold on; yline(gamma_true(1,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma1_cen,'Linewidth',2);
    ylabel('\gamma');xlabel(['iter; true = ', num2str(gamma_true(1,1)), ', est = ', num2str(gamma1_cen)]);xlim([0 nsave]);
    
    F1{2,2} = figure('visible','off');
    plot(1:nsave,gamma2_save(:,1)); hold on; yline(gamma_true(2,1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(gamma2_cen,'Linewidth',2);
    ylabel('\gamma');xlabel(['iter; true = ', num2str(gamma_true(2,1)), ', est = ', num2str(gamma2_cen)]);xlim([0 nsave]);
end

% omega
F1{3,1} = figure('visible','off');
subplot(3,1,1)
plot(1:nsave,omega_save(:,1)); hold on; yline(omega_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_cen_mix(1),'Linewidth',2);
ylabel('\omega_{1}, g=1');xlabel(['iter; true = ', num2str(omega_true(1)), ', est = ', num2str(omega_cen_mix(1))]);xlim([0 nsave]);
subplot(3,1,2)
plot(1:nsave,omega_save(:,2)); hold on; yline(omega_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_cen_mix(2),'Linewidth',2);
ylabel('\omega_{2}, g=2');xlabel(['iter; true = ', num2str(omega_true(2)), ', est = ', num2str(omega_cen_mix(2))]);xlim([0 nsave]);
subplot(3,1,3)
plot(1:nsave,omega_save(:,3)); hold on; yline(omega_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_cen_mix(3),'Linewidth',2);
ylabel('\omega_{3}, g=3');xlabel(['iter; true = ', num2str(omega_true(3)), ', est = ', num2str(omega_cen_mix(3))]);xlim([0 nsave]);

F1{3,2} = figure('visible','off');
plot(1:nsave,omega_save(:,1)); 
hold on;
plot(1:nsave,omega_save(:,2)); 
hold on;
plot(1:nsave,omega_save(:,3)); 
ylabel('\omega');xlabel('iter');
legend('\omega_{1}','\omega_{2}','\omega_{3}');

% sigmasq_u
F1{4,1} = figure('visible','off');
subplot(3,2,1)
plot(1:nsave,sigmasq_v_save(:,1)); hold on; yline(sigmasq_v_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_v_cen_mix(1),'Linewidth',2);
ylabel('\sigma^2_v_{1}, g=1');xlabel(['iter; true = ', num2str(sigmasq_v_true(1)), ', est = ', num2str(sigmasq_v_cen_mix(1))]);xlim([0 nsave]);
subplot(3,2,3)
plot(1:nsave,sigmasq_v_save(:,2)); hold on; yline(sigmasq_v_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_v_cen_mix(2),'Linewidth',2);
ylabel('\sigma^2_v_{2}, g=2');xlabel(['iter; true = ', num2str(sigmasq_v_true(2)), ', est = ', num2str(sigmasq_v_cen_mix(2))]);xlim([0 nsave]);
subplot(3,2,5)
plot(1:nsave,sigmasq_v_save(:,3)); hold on; yline(sigmasq_v_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_v_cen_mix(3),'Linewidth',2);
ylabel('\sigma^2_v_{3}, g=3');xlabel(['iter; true = ', num2str(sigmasq_v_true(3)), ', est = ', num2str(sigmasq_v_cen_mix(3))]);xlim([0 nsave]);
subplot(3,2,2)
plot(1:nsave,sigmasq_u_save(:,1)); hold on; yline(sigmasq_u_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_u_cen_mix(1),'Linewidth',2);
ylabel('\sigma^2_u_{1}, g=1');xlabel(['iter; true = ', num2str(sigmasq_u_true(1)), ', est = ', num2str(sigmasq_u_cen_mix(1))]);xlim([0 nsave]);
subplot(3,2,4)
plot(1:nsave,sigmasq_u_save(:,2)); hold on; yline(sigmasq_u_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_u_cen_mix(2),'Linewidth',2);
ylabel('\sigma^2_u_{2}, g=2');xlabel(['iter; true = ', num2str(sigmasq_u_true(2)), ', est = ', num2str(sigmasq_u_cen_mix(2))]);xlim([0 nsave]);
subplot(3,2,6)
plot(1:nsave,sigmasq_u_save(:,3)); hold on; yline(sigmasq_u_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_u_cen_mix(3),'Linewidth',2);
ylabel('\sigma^2_u_{3}, g=3');xlabel(['iter; true = ', num2str(sigmasq_u_true(3)), ', est = ', num2str(sigmasq_u_cen_mix(3))]);xlim([0 nsave]);

F1{4,2} = figure('visible','off');
plot(1:nsave,sigmasq_u_save(:,1)); 
hold on;
plot(1:nsave,sigmasq_u_save(:,2)); 
hold on;
plot(1:nsave,sigmasq_u_save(:,3)); 
ylabel('\sigma^2_u');xlabel('iter');
legend('\sigma^2_u_{1}','\sigma^2_u_{2}','\sigma^2_u_{3}')

% sigma_uv
F1{5,1} = figure('visible','off');
subplot(3,1,1)
plot(1:nsave,cov_uv_save(:,1)); hold on; yline(cov_uv_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(cov_uv_cen_mix(1),'Linewidth',2);
ylabel('\sigma_{uv,1}, g=1');xlabel(['iter; true = ', num2str(cov_uv_true(1)), ', est = ', num2str(cov_uv_cen_mix(1))]);xlim([0 nsave]);
subplot(3,1,2)
plot(1:nsave,cov_uv_save(:,2)); hold on; yline(cov_uv_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(cov_uv_cen_mix(2),'Linewidth',2);
ylabel('\sigma_{uv,2}, g=2');xlabel(['iter; true = ', num2str(cov_uv_true(2)), ', est = ', num2str(cov_uv_cen_mix(2))]);xlim([0 nsave]);
subplot(3,1,3)
plot(1:nsave,cov_uv_save(:,3)); hold on; yline(cov_uv_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(cov_uv_cen_mix(3),'Linewidth',2);
ylabel('\sigma_{uv3}, g=3');xlabel(['iter; true = ', num2str(cov_uv_true(3)), ', est = ', num2str(cov_uv_cen_mix(3))]);xlim([0 nsave]);

F1{5,2} = figure('visible','off');
plot(1:nsave,cov_uv_save(:,1)); 
hold on;
plot(1:nsave,cov_uv_save(:,2)); 
hold on;
plot(1:nsave,cov_uv_save(:,3)); 
ylabel('\sigma^2_u');xlabel('iter');
legend('\sigma_{uv,1}','\sigma_{uv,2}','\sigma_{uv,3}')

%% Set names and save graphs
n_coef = 5;
n_fig  = 2;

for i_coef = 1:n_coef
    for i_fig = 1:n_fig
        graph_tag = [result_subpath,'/',coef_names{i_coef},'_', fig_names{i_fig}];
        saveas(F1{i_coef,i_fig},graph_tag,'jpg')
    end % i_fig
end % n_coef

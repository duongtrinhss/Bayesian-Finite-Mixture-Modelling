% Generate Graphs: Convergence Diagnostics

%% Preliminary
addpath('tools');

%% Input
% date = string(datetime("today"));
% date = strrep(date,"-","");
date = '29Jul2023';
result_path = ['Results_FiniteMixture_MLR' '_' date];
mkdir(result_path);

%% Setup style
n_coef = 3;
n_fig  = 2;
F1 = cell(n_coef,n_fig);

coef_names   = {'beta','sigmasq','omega'};
fig_names    = {'trace','tracemerge'};


%% Load data, overall results, and estimator-specific results

spec = ['DGP',num2str(options.form),'/','N',num2str(n),'_','Heter',num2str(options.heter),'_','R2y',num2str(extout.R2*100)];
method = ['BayesianMixtureMLR_gunknown'];

result_subpath = [result_path,'/',spec,'/',method];
mkdir(result_subpath)


if (opt==1)
omega_save = resMedian.omega_save;
beta1_save = resMedian.beta1_save;
beta2_save = resMedian.beta2_save;
sigmasq_save = resMedian.sigmasq_save;

omega_cen_mix = resMedian.omega_median_mix;
beta1_cen_mix = resMedian.beta1_median_mix;
beta2_cen_mix = resMedian.beta2_median_mix;
sigmasq_cen_mix = resMedian.sigmasq_median_mix;

elseif (opt==2)
omega_save = resMean.omega_save;
beta1_save = resMean.beta1_save;
beta2_save = resMean.beta2_save;
sigmasq_save = resMean.sigmasq_save;

omega_cen_mix = resMean.omega_mean_mix;
beta1_cen_mix = resMean.beta1_mean_mix;
beta2_cen_mix = resMean.beta2_mean_mix;
sigmasq_cen_mix = resMean.sigmasq_mean_mix;
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


% sigmasq
F1{2,1} = figure('visible','off');
subplot(3,1,1)
plot(1:nsave,sigmasq_save(:,1)); hold on; yline(sigmasq_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_cen_mix(1),'Linewidth',2);
ylabel('\sigma^2_{1}, g=1');xlabel(['iter; true = ', num2str(sigmasq_true(1)), ', est = ', num2str(sigmasq_cen_mix(1))])
subplot(3,1,2)
plot(1:nsave,sigmasq_save(:,2)); hold on; yline(sigmasq_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_cen_mix(2),'Linewidth',2);
ylabel('\sigma^2_{2}, g=2');xlabel(['iter; true = ', num2str(sigmasq_true(2)), ', est = ', num2str(sigmasq_cen_mix(2))])
subplot(3,1,3)
plot(1:nsave,sigmasq_save(:,3)); hold on; yline(sigmasq_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(sigmasq_cen_mix(3),'Linewidth',2);
ylabel('\sigma^2_{3}, g=3');xlabel(['iter; true = ', num2str(sigmasq_true(3)), ', est = ', num2str(sigmasq_cen_mix(3))])

F1{2,2} = figure('visible','off');
plot(1:nsave,sigmasq_save(:,1)); 
hold on;
plot(sigmasq_save(:,2)); 
hold on;
plot(sigmasq_save(:,3)); 
ylabel('\sigma^2');xlabel('iter');
legend('\sigma^2_{1}','\sigma^2_{2}','\sigma^2_{3}');

% omega
F1{3,1} = figure('visible','off');
subplot(3,1,1)
plot(1:nsave,omega_save(:,1)); hold on; yline(omega_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_cen_mix(1),'Linewidth',2);
ylabel('\omega_{1}, g=1');xlabel(['iter; true = ', num2str(omega_true(1)), ', est = ', num2str(omega_cen_mix(1))])
subplot(3,1,2)
plot(1:nsave,omega_save(:,2)); hold on; yline(omega_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_cen_mix(2),'Linewidth',2);
ylabel('\omega_{2}, g=2');xlabel(['iter; true = ', num2str(omega_true(2)), ', est = ', num2str(omega_cen_mix(2))])
subplot(3,1,3)
plot(1:nsave,omega_save(:,3)); hold on; yline(omega_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(omega_cen_mix(3),'Linewidth',2);
ylabel('\omega_{3}, g=3');xlabel(['iter; true = ', num2str(omega_true(3)), ', est = ', num2str(omega_cen_mix(3))])

F1{3,2} = figure('visible','off');
plot(1:nsave,omega_save(:,1)); 
hold on;
plot(1:nsave,omega_save(:,2)); 
hold on;
plot(1:nsave,omega_save(:,3)); 
ylabel('\omega');xlabel('iter');
legend('\omega_{1}','\omega_{2}','\omega_{3}');

%% Set names and save graphs
n_coef = 3;
n_fig  = 2;

for i_coef = 1:n_coef
    for i_fig = 1:n_fig
        graph_tag = [result_subpath,'/',coef_names{i_coef},'_', fig_names{i_fig}];
        saveas(F1{i_coef,i_fig},graph_tag,'jpg')
    end % i_fig
end % n_coef
% Generate Graphs: Convergence Diagnostics

%% Preliminary
addpath('tools');

%% Input
% date = string(datetime("today"));
% date = strrep(date,"-","");
date = '13Jul2023';
result_path = ['Results_FiniteMixture_SAR' '_' date];
mkdir(result_path);

%% Setup style
n_coef = 4;
n_fig  = 2;
F1 = cell(n_coef,n_fig);

coef_names   = {'beta_perm','rho_perm','sigmasq_v_perm','omega_perm'};
% coef_names   = {'betapermbs','rho_perm_bs','sigmasqpermmbs','omegapermbs'};
fig_names    = {'trace','tracemerge'};


%% Load data, overall results, and estimator-specific results

spec = ['DGP',num2str(options.form),'/','N',num2str(N),'_','R2y',num2str(extout.R2*100)];
method = ['BayesianMixtureSARrmwh_quickadaptive_gunknown'];

result_subpath = [result_path,'/',spec,'/',method];
mkdir(result_subpath)

%% Plot graphs
% parperm.beta
F1{1,1} = figure('visible','off');
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

F1{1,2} = figure('visible','off');
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
F1{2,1} = figure('visible','off');
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
F1{3,1} = figure('visible','off');
subplot(3,1,1)
plot(1:nsave,parperm.sigmav_save(:,1)); hold on; yline(sigmasq_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmav_mean_mix(1),'Linewidth',2);
ylabel('\sigma^2_v_{1}, g=1');xlabel(['iter; true = ', num2str(sigmasq_true(1)), ', est = ', num2str(parperm.sigmav_mean_mix(1))]);xlim([0 nsave]);
subplot(3,1,2)
plot(1:nsave,parperm.sigmav_save(:,2)); hold on; yline(sigmasq_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmav_mean_mix(2),'Linewidth',2);
ylabel('\sigma^2_v_{2}, g=2');xlabel(['iter; true = ', num2str(sigmasq_true(2)), ', est = ', num2str(parperm.sigmav_mean_mix(2))]);xlim([0 nsave]);
subplot(3,1,3)
plot(1:nsave,parperm.sigmav_save(:,3)); hold on; yline(sigmasq_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.sigmav_mean_mix(3),'Linewidth',2);
ylabel('\sigma^2_v_{3}, g=3');xlabel(['iter; true = ', num2str(sigmasq_true(3)), ', est = ', num2str(parperm.sigmav_mean_mix(3))]);xlim([0 nsave]);

F1{3,2} = figure('visible','off');
plot(1:nsave,parperm.sigmav_save(:,1)); 
hold on;
plot(1:nsave,parperm.sigmav_save(:,2)); 
hold on;
plot(1:nsave,parperm.sigmav_save(:,3)); 
ylabel('\sigma^2_v');xlabel('iter');
legend('\sigma^2_v_{1}','\sigma^2_v_{2}','\sigma^2_v_{3}')

% parperm.omega
F1{4,1} = figure('visible','off');
subplot(3,1,1)
plot(1:nsave,parperm.omega_save(:,1)); hold on; yline(omega_true(1),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.omega_mean_mix(1),'Linewidth',2);
ylabel('\omega_{1}, g=1');xlabel(['iter; true = ', num2str(omega_true(1)), ', est = ', num2str(parperm.omega_mean_mix(1))]);xlim([0 nsave]);
subplot(3,1,2)
plot(1:nsave,parperm.omega_save(:,2)); hold on; yline(omega_true(2),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.omega_mean_mix(2),'Linewidth',2);
ylabel('\omega_{2}, g=2');xlabel(['iter; true = ', num2str(omega_true(2)), ', est = ', num2str(parperm.omega_mean_mix(2))]);xlim([0 nsave]);
subplot(3,1,3)
plot(1:nsave,parperm.omega_save(:,3)); hold on; yline(omega_true(3),'Linewidth',2,'Color',[.6 0 0]); hold on; yline(parperm.omega_mean_mix(3),'Linewidth',2);
ylabel('\omega_{3}, g=3');xlabel(['iter; true = ', num2str(omega_true(3)), ', est = ', num2str(parperm.omega_mean_mix(3))]);xlim([0 nsave]);

F1{4,2} = figure('visible','off');
plot(1:nsave,parperm.omega_save(:,1)); 
hold on;
plot(1:nsave,parperm.omega_save(:,2)); 
hold on;
plot(1:nsave,parperm.omega_save(:,3)); 
ylabel('\omega');xlabel('iter');
legend('\omega_{1}','\omega_{2}','\omega_{3}');

%% Set names and save graphs
n_coef = 4;
n_fig  = 2;

for i_coef = 1:n_coef
    for i_fig = 1:n_fig
        graph_tag = [result_subpath,'/',coef_names{i_coef},'_', fig_names{i_fig}];
        saveas(F1{i_coef,i_fig},graph_tag,'jpg')
    end % i_fig
end % n_coef



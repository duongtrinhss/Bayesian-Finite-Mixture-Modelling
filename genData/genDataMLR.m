function [Y,x,g,beta,sigmasq,omega,extout] = genDataMLR(n,options)

if (options.form == 1)
%--------------------------------------------------------------------------
% DGP1: 
%--------------------------------------------------------------------------
% D = gamma*Z + v
% Y = beta*X + u
% Covmat = [sigma_v^2   sigma_uv
%           sigma_uv    sigma_u^2] 

% 0. Generate component label
G     = 3;
omega = [0.45 0.35 0.2];
g     = mnrnd(1,omega,n);
mg    = sum(g);

% 1. Generate covariates
% x = randn(n,2)*sqrt(9);
% x = 2*ones(n,2) + sqrt(16).*randn(n,2);
x = [ones(n,1) randn(n,1)*sqrt(9)];

% 2. Generate slopes
if (options.heter == 0)
beta = [0.5 0.8;
            0.5  0.8;
            0.5  0.8]';
elseif (options.heter == 1)
beta = [-0.5 0.8;
            0.5  0.8;
            -1.0  0.8]';
elseif (options.heter == 2)
beta = [0.5 -0.75;
            0.5  0.8;
            0.5  1.2]'; 
elseif (options.heter == 3)
beta = [-0.5 -0.75;
            0.5  0.8;
            -1.0  1.2]'; 
end

beta_til = g*beta';

% 3. Generate errors
sigmasq = [1 0.75 0.5];

%==== Adjust c to obtain desirable options.R2 (or SNR)
signal_part = var(sum(beta_til.*x,2));

c = (signal_part/(sigmasq*omega'))*((1-options.R2)/options.R2);
sigmasq = c*sigmasq;

%========= Check
extout.R2 = signal_part/(signal_part+sigmasq*omega');

% 4. Generate depvar Y
Y = zeros(n,1);
for l=1:G
    points = find(g(:,l)==1);
    npts   = sum(g(:,l)==1);
    
    epsilon = randn(npts,1)*sqrt(sigmasq(:,l));
    Y(points,:) = x(points,:)*beta(:,l) + epsilon;
    
    clear points npts
end
end
end
function [Y,D,z,x,gamma,beta,covmat,extout] = genData2SLS(n,options)
% genData2SLS: DGP for 2SLS models
% =========================================================================                
%
% =========================================================================
% Written by Duong Trinh
% University of Glasgow
% This version: July 2023
% =========================================================================

if (options.form == 1)
%--------------------------------------------------------------------------
% DGP1: 
%--------------------------------------------------------------------------
% D = gamma*Z + v
% Y = beta*X + u
% Covmat = [sigma_v^2   sigma_uv
%           sigma_uv    sigma_u^2] 

% 1. Generate covariates
z = randn(n,1);
x = randn(n,2);

% 2. Generate slopes
gamma = [1 1.5 .8 -.5]';
beta  = [2  1  .6 -.4]';

% 3. Generate errors
corr_uv = 0.5;
sigmasq_v = 1;
sigmasq_u = 1;

%==== Adjust c to obtain desirable options.R2 (or SNR)
signal_d = norm(chol(eye(3))*gamma(2:end))^2; % var([z c]*gamma(2:end))  
signal_y = norm(chol(eye(2))*beta(3:end))^2;  % var([c]*beta(3:end))

c_d = (signal_d/sigmasq_v)*((1-options.R2_d)/options.R2_d);
c_y = (signal_y/sigmasq_u)*((1-options.R2_y)/options.R2_y);

sigmasq_v = c_d*sigmasq_v;
sigmasq_u = c_y*sigmasq_u;
cov_uv = corr_uv*sqrt(sigmasq_v*sigmasq_u);
covmat = [sigmasq_v  cov_uv;
            cov_uv sigmasq_u];

%========= Check
extout.R2_d = signal_d/(signal_d+sigmasq_v);
extout.R2_y = signal_y/(signal_y+sigmasq_u);

epsilon = randn(n,2)*chol(covmat);

% 4. Generate D and Y

zmat = [ones(n,1) z x];
D    =  zmat*gamma + epsilon(:,1);

xmat = [ones(n,1) D x];
Y    = xmat*beta + epsilon(:,2);

elseif (options.form == 2)
%--------------------------------------------------------------------------
% DGP2: mixture Gaussian
%--------------------------------------------------------------------------
% D = gamma*Z + vg
% Y = beta*X + ug
% Covmat = [sigma_vg^2   sigma_uvg
%           sigma_uvg    sigma_ug^2] 

% 0. Generate component label
G     = 3;
omega = [0.45 0.35 0.2];
g     = mnrnd(1,omega,n);
mg    = sum(g);

%==== Export 'g' and 'omega' as extra output
extout.g     = g;
extout.omega = omega; 

% 1. Generate covariates
z = randn(n,1)*sqrt(4);
x = randn(n,2)*sqrt(9);

pd = size([z x],2); % number of covariates in D-equation
py = size([x],2);   % number of covariates (except for D) in Y-equation

% 2. Generate slopes
gamma = [1 1.5 .8 -.5;
            1 1.5 .8 -.5;
            1 1.5 .8 -.5]';
beta  = [2  1  .6 -.4;
            2  1  .6 -.4;
            2  1  .6 -.4]';

gamma_til = g*gamma';
beta_til  = g*beta';

% 3. Generate errors
corr_uv = [0.5 0.5 0.5];
sigmasq_u = [1 0.75 0.5];
sigmasq_v = [1 0.75 0.5];

%==== Adjust c to obtain desirable options.R2 (or SNR)
signal_d = var(sum(gamma_til(:,2:end).*[z x],2));
signal_y = var(sum(beta_til(:,3:end).*x,2));

c_d = (signal_d/(sigmasq_v*omega'))*((1-options.R2_d)/options.R2_d);
c_y = (signal_y/(sigmasq_u*omega'))*((1-options.R2_y)/options.R2_y);

sigmasq_v = c_d*sigmasq_v;
sigmasq_u = c_y*sigmasq_u;
cov_uv    = corr_uv.*sqrt(sigmasq_v.*sigmasq_u);

%========= Check
extout.R2_d = signal_d/(signal_d+sigmasq_v*omega');
extout.R2_y = signal_y/(signal_y+sigmasq_u*omega');

covmat = reshape([sigmasq_v;
    cov_uv;
    cov_uv;
    sigmasq_u],2,2,3);

D = zeros(n,1);
Y = zeros(n,1);

for l=1:G
    points = find(g(:,l)==1);
    npts   = sum(g(:,l)==1);
    
    epsilon = randn(npts,2)*chol(covmat(:,:,l));
    D(points,:) = [ones(npts,1) z(points,:) x(points,:)]*gamma(:,l) + epsilon(:,1);
    Y(points,:) = [ones(npts,1) D(points,:) x(points,:)]*beta(:,l) + epsilon(:,2);
    
    clear points npts
end

elseif (options.form == 3)
%--------------------------------------------------------------------------
% DGP3: 
%--------------------------------------------------------------------------
% D = gamma*Z + f
% Y = beta*X + f + u
% Covmat = [sigma_v^2   sigma_uv
%           sigma_uv    sigma_u^2] 

% 1. Generate covariates
z = randn(n,1);
x = randn(n,2);

% 2. Generate slopes
gamma = [1 1.5 .8 -.5]';
beta  = [2  1  .6 -.4]';

% 3. Generate unobserved distribution
f = randn(n,1);
lambD = 0.8;
lambY = 0.3;

sigmasq_v = 1;
sigmasq_u = 0.25;

epsD = lambD*f + sqrt(sigmasq_v)*randn(n,1);
epsY = lambY*f + sqrt(sigmasq_u)*randn(n,1);

covmat = [sigmasq_v + lambD'*lambD  lambY'*lambD;
            lambD'*lambY sigmasq_u + lambY'*lambY];
        
epsilon = [epsD epsY];

%==== Export 'lambD' and 'lambY' as extra output
extout.lambD = lambD;
extout.lambY = lambY;
extout.f = f;


% 4. Generate D and Y
zmat = [ones(n,1) z x];
D    =  zmat*gamma + epsilon(:,1);

xmat = [ones(n,1) D x];
Y    = xmat*beta + epsilon(:,2);

elseif (options.form == 4)
%--------------------------------------------------------------------------
% DGP4: 
%--------------------------------------------------------------------------
% D = gamma*Z + v
% Y = delta*D + beta*X + u
% Covmat = [sigma_v^2   sigma_uv
%           sigma_uv    sigma_u^2] 

% 1. Generate covariates
z = randn(n,1)*sqrt(4);
x = randn(n,2)*sqrt(9);

% 2. Generate slopes
gamma = [1 1.5 .8 -.5]';
beta  = [2  1  .6 -.4]';

% 3. Generate errors
corr_uv = 0.5;
sigmasq_v = 1;
sigmasq_u = 1.25;

%==== Adjust c to obtain desirable options.R2 (or SNR)
signal_d = var([z x]*gamma(2:end));  
signal_y = var([x]*beta(3:end));

% c_d = (signal_d/sigmasq_v)*((1-options.R2_d)/options.R2_d);
% c_y = (signal_y/sigmasq_u)*((1-options.R2_y)/options.R2_y);

c_d = 1;
c_y = 1;

sigmasq_v = c_d*sigmasq_v;
sigmasq_u = c_y*sigmasq_u;
cov_uv = corr_uv*sqrt(sigmasq_v*sigmasq_u);
covmat = [sigmasq_v  cov_uv;
            cov_uv sigmasq_u];

%========= Check
extout.R2_d = signal_d/(signal_d+sigmasq_v);
extout.R2_y = signal_y/(signal_y+sigmasq_u);

epsilon = randn(n,2)*chol(covmat);

% 4. Generate D and Y

zmat = [ones(n,1) z x];
D    =  zmat*gamma + epsilon(:,1);

xmat = [ones(n,1) D x];
Y    = xmat*beta + epsilon(:,2);

elseif (options.form == 5)
%--------------------------------------------------------------------------
% DGP5: 
%--------------------------------------------------------------------------
% D = gamma*Z + v
% Y = deltag*D + betag*X + ug
% Covmat = [1   sigma_uvg
%           sigma_uvg    sigma_ug^2] 

% 0. Generate component label
G     = 3;
omega = [0.45 0.35 0.2];
g     = mnrnd(1,omega,n);
mg    = sum(g);

%==== Export 'g' and 'omega' as extra output
extout.g     = g;
extout.omega = omega; 

% 1. Generate covariates
z = randn(n,1)*sqrt(4);
x = randn(n,2)*sqrt(9);

% 2. Generate slopes
gamma = [1 1.5 .8 -.5;
            1 1.5 .8 -.5;
            1 1.5 .8 -.5]';
beta  = [2  1.5  .6 -.4;
            2  1  .6 -.4;
            2  -1  .6 -.4]';

gamma_til = g*gamma';
beta_til  = g*beta';

% 3. Generate errors
corr_uv = [0.5 0.75 0.25];
sigmasq_v = [1 1 1];
sigmasq_u = [1.25 0.8 0.5];

%==== Adjust c to obtain desirable options.R2 (or SNR)
signal_d = var(sum(gamma_til(:,2:end).*[z x],2));
signal_y = var(sum(beta_til(:,3:end).*x,2));

c_d = 1; %default
c_y = 1; %default
if ( isfield(options,'R2_d')&&isfield(options,'R2_y') )
    c_d = (signal_d/(sigmasq_v*omega'))*((1-options.R2_d)/options.R2_d);
    c_y = (signal_y/(sigmasq_u*omega'))*((1-options.R2_y)/options.R2_y);
end

sigmasq_v = c_d*sigmasq_v;
sigmasq_u = c_y*sigmasq_u;
cov_uv    = corr_uv.*sqrt(sigmasq_v.*sigmasq_u);

%========= Check
extout.R2_d = signal_d/(signal_d+sigmasq_v*omega');
extout.R2_y = signal_y/(signal_y+sigmasq_u*omega');

covmat = reshape([sigmasq_v;
    cov_uv;
    cov_uv;
    sigmasq_u],2,2,3);

D = zeros(n,1);
Y = zeros(n,1);

for l=1:G
    points = find(g(:,l)==1);
    npts   = sum(g(:,l)==1);
    
    epsilon = randn(npts,2)*chol(covmat(:,:,l));
    D(points,:) = [ones(npts,1) z(points,:) x(points,:)]*gamma(:,l) + epsilon(:,1);
    Y(points,:) = [ones(npts,1) D(points,:) x(points,:)]*beta(:,l) + epsilon(:,2);
    
    clear points npts
end

elseif (options.form == 6)
%--------------------------------------------------------------------------
% DGP6: 
%--------------------------------------------------------------------------
% D = gamma*Z + v
% Y = deltag*D + betag*X + ug
% Covmat = [1   sigma_uvg
%           sigma_uvg    sigma_ug^2] 

% 0. Generate component label
G     = 3;
omega = [0.45 0.35 0.2];
g     = mnrnd(1,omega,n);
mg    = sum(g);

%==== Export 'g' and 'omega' as extra output
extout.g     = g;
extout.omega = omega; 

% 1. Generate covariates
z = randn(n,1)*sqrt(4);
x = randn(n,2)*sqrt(16);

% 2. Generate slopes
gamma = [1 1.5 .8 -.5;
            1 1.5 .8 -.5;
            1 1.5 .8 -.5]';

%==== Consider homogeneous & heterogenous treatment effects
if (options.heter == 0)
    beta  = [2  1  .6 -.4;
        2  1  .6 -.4;
        2  1  .6 -.4]';
    
elseif (options.heter == 1)
    beta  = [2  1.5  .6 -.4;
        2  1  .6 -.4;
        2  -1  .6 -.4]';
elseif (options.heter == 2)
    beta  = [2.5  1.5  .6 -.4;
        -1.5  1  .6 -.4;
        2  -1  .6 -.4]';
end

gamma_til = g*gamma';
beta_til  = g*beta';

% 3. Generate errors
corr_uv = [0.5 0.75 0.25];
sigmasq_v = [1 1 1];         %fixed
sigmasq_u = [1.25 0.8 0.5];

%==== Adjust c to obtain desirable options.R2 (or SNR)
signal_d = var(sum(gamma_til(:,2:end).*[z x],2));
signal_y = var(sum(beta_til(:,3:end).*x,2));

c_d = 1; %default
c_y = 1; %default

if ( isfield(options,'R2_y') )
    c_y = (signal_y/(sigmasq_u*omega'))*((1-options.R2_y)/options.R2_y);
end

sigmasq_v = c_d*sigmasq_v;
sigmasq_u = c_y*sigmasq_u;
cov_uv    = corr_uv.*sqrt(sigmasq_v.*sigmasq_u);

%========= Check
extout.R2_d = signal_d/(signal_d+sigmasq_v*omega');
extout.R2_y = signal_y/(signal_y+sigmasq_u*omega');

covmat = reshape([sigmasq_v;
    cov_uv;
    cov_uv;
    sigmasq_u],2,2,3);

% 4. Generate D and Y
D = zeros(n,1);
Y = zeros(n,1);

zmat = [ones(n,1) z x];

epsD = randn(n,1);
D    =  zmat*gamma(:,1) + epsD;

%========= Export 'epsD' as extra output
extout.epsD = epsD;

for l=1:G
    points = find(g(:,l)==1);
    npts   = sum(g(:,l)==1);
        
    epsY = cov_uv(:,l)*epsD(points,:) + sqrt(sigmasq_u(:,l) - cov_uv(:,l)^2)*randn(npts,1);
    
    Y(points,:) = [ones(npts,1) D(points,:) x(points,:)]*beta(:,l) + epsY;
    
    clear points npts
end
elseif (options.form == 7)
%--------------------------------------------------------------------------
% DGP7: 
%--------------------------------------------------------------------------
% D* = gamma*Z + v; D = 1{D*>=0}
% Y = delta*D + beta*X + u
% Covmat = [1           sigma_uv
%           sigma_uv    sigma_u^2] 

% 1. Generate covariates
z = randn(n,1)*sqrt(4);
x = randn(n,2)*sqrt(16);

% 2. Generate slopes
gamma = [1 1.5 .8 -.5]';
beta  = [2  1  .6 -.4]';

if ( isfield(options,'group') )
    
    if (options.group == 1)
        beta  = [2.5  1.5  .6 -.4]';
    elseif (options.group == 2)
        beta  = [-1.5  1  .6 -.4]';
    elseif (options.group == 3)
        beta  = [2  -1  .6 -.4]';
    end
    
end
% 3. Generate errors
corr_uv = 0.5;
sigmasq_v = 1;
sigmasq_u = 1.25;

cov_uv = corr_uv*sqrt(sigmasq_v*sigmasq_u);
covmat = [sigmasq_v  cov_uv;
            cov_uv sigmasq_u];

epsilon = randn(n,2)*chol(covmat);

% 4. Generate D and Y
zmat   = [ones(n,1) z x];
D_star =  zmat*gamma + epsilon(:,1);
D = 1*(D_star>0);

xmat = [ones(n,1) D x];
Y    = xmat*beta + epsilon(:,2);

%========= Check
signal_d = var([z x]*gamma(2:end));  
signal_y = var([x]*beta(3:end));
extout.R2_d = signal_d/(signal_d+sigmasq_v);
extout.R2_y = signal_y/(signal_y+sigmasq_u);

elseif (options.form == 8)
%--------------------------------------------------------------------------
% DGP8: 
%--------------------------------------------------------------------------
% D* = gamma*Z + v; D = 1{D*>=0}
% Y = deltag*D + betag*X + ug
% Covmat = [1   sigma_uvg
%           sigma_uvg    sigma_ug^2] 

% 0. Generate component label
G     = 3;
omega = [0.45 0.35 0.2];
g     = mnrnd(1,omega,n);
mg    = sum(g);

%==== Export 'g' and 'omega' as extra output
extout.g     = g;
extout.omega = omega; 

% 1. Generate covariates
z = randn(n,1)*sqrt(4);
x = randn(n,2)*sqrt(16);

% 2. Generate slopes
gamma = [1 1.5 .8 -.5;
            1 1.5 .8 -.5;
            1 1.5 .8 -.5]';

%==== Consider homogeneous & heterogenous treatment effects
if (options.heter == 0)
    beta  = [2  1  .6 -.4;
        2  1  .6 -.4;
        2  1  .6 -.4]'; 
elseif (options.heter == 1)
    beta  = [2  1.5  .6 -.4;
        2  1  .6 -.4;
        2  -1  .6 -.4]';
elseif (options.heter == 2)
    beta  = [2.5  1.5  .6 -.4;
        -1.5  1  .6 -.4;
        2  -1  .6 -.4]';
end

gamma_til = g*gamma';
beta_til  = g*beta';

% 3. Generate errors
corr_uv = [0.5 0.75 0.25];
sigmasq_v = [1 1 1];
sigmasq_u = [1.25 0.8 0.5];

%==== Adjust c to obtain desirable options.R2 (or SNR)
signal_d = var(sum(gamma_til(:,2:end).*[z x],2));
signal_y = var(sum(beta_til(:,3:end).*x,2));

c_d = 1; %fixed
c_y = 1; %fixed

% c_d = (signal_d/(sigmasq_v*omega'))*((1-options.R2_d)/options.R2_d);
% c_y = (signal_y/(sigmasq_u*omega'))*((1-options.R2_y)/options.R2_y);

sigmasq_v = c_d*sigmasq_v;
sigmasq_u = c_y*sigmasq_u;
cov_uv    = corr_uv.*sqrt(sigmasq_v.*sigmasq_u);

%========= Check
extout.R2_d = signal_d/(signal_d+sigmasq_v*omega');
extout.R2_y = signal_y/(signal_y+sigmasq_u*omega');

covmat = reshape([sigmasq_v;
    cov_uv;
    cov_uv;
    sigmasq_u],2,2,3);

% 4. Generate D and Y
D = zeros(n,1);
Y = zeros(n,1);

zmat = [ones(n,1) z x];

epsD = randn(n,1);
Dstar  =  zmat*gamma(:,1) + epsD;
D = 1*(Dstar>0);

%======== Export 'epsD' and 'Dstar' as extra output
extout.epsD = epsD;
extout.Dstar = Dstar;

for l=1:G
    points = find(g(:,l)==1);
    npts   = sum(g(:,l)==1);
        
    epsY = cov_uv(:,l)*epsD(points,:) + sqrt(sigmasq_u(:,l) - cov_uv(:,l)^2)*randn(npts,1);
    
    Y(points,:) = [ones(npts,1) D(points,:) x(points,:)]*beta(:,l) + epsY;
    
    clear points npts
end
elseif (options.form == 9)
%--------------------------------------------------------------------------
% DGP9: 
%--------------------------------------------------------------------------
% D = gamma*Z + v
% Y = deltag*D + betag*X + ug
% Covmat = [1   sigma_uvg
%           sigma_uvg    sigma_ug^2] 

% 0. Generate component label
G     = 3;
omega = [0.45 0.35 0.2];
g     = mnrnd(1,omega,n);
mg    = sum(g);

%==== Export 'g' and 'omega' as extra output
extout.g     = g;
extout.omega = omega; 

% 1. Generate covariates
z = randn(n,1)*sqrt(4);

% 2. Generate slopes
gamma = [1 1.5;
            1 1.5;
            1 1.5]';

%==== Consider homogeneous & heterogenous treatment effects
if (options.heter == 0)
    beta  = [2  1;
        2  1;
        2  1]';
    
elseif (options.heter == 1)
    beta  = [2  1.5;
        2  1;
        2  -1]';
elseif (options.heter == 2)
    beta  = [2.5  1.5;
        -1.5  1;
        2  -1]';
end

gamma_til = g*gamma';
beta_til  = g*beta';

% 3. Generate errors
corr_uv = [0.5 0.75 0.25];
sigmasq_v = [1 1 1];         %fixed
sigmasq_u = [1.25 0.8 0.5];

% %==== Adjust c to obtain desirable options.R2 (or SNR)
% signal_d = var(sum(gamma_til(:,2:end).*[z x],2));
% signal_y = var(sum(beta_til(:,3:end).*x,2));

c_d = 1; %default
c_y = 1; %default

if ( isfield(options,'R2_y') )
    c_y = (signal_y/(sigmasq_u*omega'))*((1-options.R2_y)/options.R2_y);
end

sigmasq_v = c_d*sigmasq_v;
sigmasq_u = c_y*sigmasq_u;
cov_uv    = corr_uv.*sqrt(sigmasq_v.*sigmasq_u);

% %========= Check
% extout.R2_d = signal_d/(signal_d+sigmasq_v*omega');
% extout.R2_y = signal_y/(signal_y+sigmasq_u*omega');

covmat = reshape([sigmasq_v;
    cov_uv;
    cov_uv;
    sigmasq_u],2,2,3);

% 4. Generate D and Y
D = zeros(n,1);
Y = zeros(n,1);

zmat = [ones(n,1) z];

epsD = randn(n,1);
D    =  zmat*gamma(:,1) + epsD;

%========= Export 'epsD' as extra output
extout.epsD = epsD;

for l=1:G
    points = find(g(:,l)==1);
    npts   = sum(g(:,l)==1);
        
    epsY = cov_uv(:,l)*epsD(points,:) + sqrt(sigmasq_u(:,l) - cov_uv(:,l)^2)*randn(npts,1);
    
    Y(points,:) = [ones(npts,1) D(points,:)]*beta(:,l) + epsY;
    
    clear points npts
end

x=[];

end
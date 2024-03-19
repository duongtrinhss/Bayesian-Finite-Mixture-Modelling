function [Y,Wr,X,beta,rho,sigmasq,extout]=genDataSAR(N,options)

if (options.form==1) % standard SAR && exogenous spatial setting
%--------------------------------------------------------------------------
% DGP1: 
%-------------------------------------------------------------------------- 
% 1. Generate spatial weight matrice
geos = 0.5*ones(N,2) + sqrt(0.01).*randn(N,2);
xc = geos(:,1);
yc = geos(:,2);

%==== Create contiguity matrix from x-y coordinates
[W1 W2 W3] = xy2cont(xc,yc);
Wr = W2; % row-normalized matrix

% 2. Generate coefficients
beta = [0.5 0.8]';
rho = 0.3;

% 3. Generate covariates X
% X = randn(N,2)*sqrt(9);
X = 2*ones(N,2) + sqrt(16).*randn(N,2);
% X = [ones(N,1) randn(N,1)*sqrt(9)];

% 4. Generate disturbance
sigmasq = 1.25;
eps = sqrt(sigmasq)*randn(N,1);

%===== Adjust c to obtain desirable options.R2 (or SNR)
signal_part = beta'*X'*( speye(N)-rho*Wr )^(-2)*X*beta;
noise_part  = (sigmasq)*trace(( speye(N)-rho*Wr )^(-2));

c = 1; %default
if (isfield(options,'R2'))
    c = (signal_part/noise_part)*((1-options.R2)/options.R2);
    sigmasq = c*sigmasq;
    noise_part  = (sigmasq)*trace(( speye(N)-rho*Wr )^(-2));
end

%========= Check
extout.R2 = signal_part/(signal_part+noise_part);

% 5. Generate depvar Y
S = ( speye(N)-rho*Wr )^(-1);
Y = S*X*beta + S*eps;
    
%==========================================================================
elseif (options.form==2) % mixture SAR && exogenous spatial setting
%--------------------------------------------------------------------------
% DGP2: 
%--------------------------------------------------------------------------
% 0. Generate component label
G = 3;
omega = [0.45; 0.35; 0.2];
g = mnrnd(1,omega,N);
mg = sum(g);
%==== Export 'g' and 'omega' as extra output
extout.g     = g;
extout.omega = omega; 

% 1. Generate spatial weight matrice
geos = zeros(N,2);
geos(g(:,1)==1,:) = 0.5*ones(mg(1),2) + sqrt(0.01).*randn(mg(1),2);
geos(g(:,2)==1,:) = 1*ones(mg(2),2) + sqrt(0.01).*randn(mg(2),2);
geos(g(:,3)==1,:) = 1.5*ones(mg(3),2) + sqrt(0.01).*randn(mg(3),2);

xc = geos(:,1);
yc = geos(:,2);
cl = 1*g(:,1) + 2*g(:,2) + 3*g(:,3);

%==== Create contiguity matrix from x-y coordinates
[W1 W2 W3] = xy2cont(xc,yc);
Wr = W2; % row-normalized matrix

% 2. Generate coefficients
beta = [-0.5 -0.75;
            0.5  0.8;
            -1.0  1.2]';

rho = [-0.3 0.3 0.6];

beta_til = g*beta';
rho_til  = g*rho';

% beta_vec = reshape(beta,k*G,1);

% 3. Generate covariates
% X = randn(N,2)*sqrt(9);
X = 2*ones(N,2) + sqrt(16).*randn(N,2);
% X = [ones(N,1) randn(N,1)*sqrt(9)];

% 4. Generate errors
sigmasq = [1.0 0.75 0.5];

%==== Adjust c to obtain desirable options.R2 (or SNR)
signal_part = (sum(beta_til.*X,2))'*( speye(N)-diag(rho_til)*Wr )^(-2)*(sum(beta_til.*X,2));
noise_part  = (sigmasq*omega)*trace(( speye(N)-diag(rho_til)*Wr )^(-2));

c = 1; %default
if (isfield(options,'R2'))
    c = (signal_part/noise_part)*((1-options.R2)/options.R2);
    sigmasq = c*sigmasq;
    noise_part  = (sigmasq*omega)*trace(( speye(N)-diag(rho_til)*Wr )^(-2));
end

%========= Check
extout.R2 = signal_part/(signal_part+noise_part);

%==== Epsilon
eps     = randn(N,1);
eps_til = (g*sqrt(sigmasq')).*eps;

% 5. Generate depvar Y
S = ( speye(N)-diag(rho_til)*Wr )^(-1);
Y = S*sum(beta_til.*X,2) + S*eps_til;

%==========================================================================
elseif (options.form==3)      % standard SAR && exogenous network setting
%%-------------------------------------------------------------------------
% DGP3: 
%%-------------------------------------------------------------------------  
% 1. Generate explanatory variables
%===== for  W-Equation
c_1 = rand([N,1]); 
c1  = rand([N,N]); 
for i=1:N 
    for j=1:N 
        if c_1(i,:)>=0.7 && c_1(j,:)>=0.7 
            c1(i,j)=1; 
        elseif c_1(i,:)<=0.3 && c_1(j,:)<=0.3 
            c1(i,j)=1; 
        else 
            c1(i,j)=0; 
        end 
        if i==j 
            c1(i,j)=0; 
        end 
    end 
end 

C = [c1]; 

%===== for  Y-Equation
% x = randn(N,2)*sqrt(9);
x = 2*ones(N,2) + sqrt(16).*randn(N,2);
X = [x]; 

% 2. Generate coefficients
%===== for  W-Equation
gamma   = [1.5]; 

%===== for  Y-Equation
beta    = [0.5; 0.8]; 
rho  = 0.3;

% 3. Generate covariance matrix
Sigma = [1 0; 
         0 1.25]; % corr.coef = 0 - exogenous network setting
mu      = [0.5 0]; 
sigmasq = Sigma(2,2); 
kappa   = Sigma(1,2); 

a   = mu(1) + sqrt(Sigma(1,1))*randn(N,1); % shared latent variable
u   = mu(2) + sqrt(sigmasq)*randn(N,1); % error term in Y-Equation

% 4. W-Equation (Network Formation)

p = zeros(N,N); % probability of forming a link
w = zeros(N,N); % network link
for i=1:N 
    for j=1:N 
        psi= gamma(1)*c1(i,j) + a(i,1) + a(j,1); 
        switch options.nwdist 
            case 'logit' 
                p(i,j)=exp(psi)/(1+exp(psi));      % Logit setting 
            case 'probit' 
                p(i,j) = normcdf(psi,0,1);         % Probit setting 
        end 
        if rand(1)<=p(i,j) 
            w(i,j)=1; 
        else 
            w(i,j)=0; 
        end 
        if i==j 
            w(i,j)=0; 
        end 
    end 
end 
%===== Network Matrix
W = w;     % save 0-1 matrix 
%========== Normalize the rows 
Wmat = W; 
ridx = find(sum(Wmat, 2)~=0); 
Wmat(ridx,:) = bsxfun(@rdivide, Wmat(ridx,:), sum(Wmat(ridx,:), 2)); 
Wr = Wmat; % save row-normalised matrix 

% 5. Y-Equation (Network Interaction - SAR)
S = ( speye(N)-rho*Wr )^(-1); 
Y = S*(X*beta + kappa*a + u); 

%==== Adjust c to obtain desirable options.R2 (or SNR)
signal_part = [beta; kappa]'*[X a]'*( speye(N)-rho*Wr )^(-2)*[X a]*[beta; kappa];
noise_part  = (sigmasq)*trace(( speye(N)-rho*Wr )^(-2));

c = 1; %default
if (isfield(options,'R2'))
    c = (signal_part/noise_part)*((1-options.R2)/options.R2);
    sigmasq = c*sigmasq;
    noise_part  = (sigmasq)*trace(( speye(N)-rho*Wr )^(-2));
    u   = mu(2) + sqrt(sigmasq)*randn(N,1); % adjust error term in Y-Equation
end
%========= Check
extout.R2   = signal_part/(signal_part+noise_part);

%==========================================================================
elseif (options.form==4)      % mixture SAR && exogenous network setting
%%-------------------------------------------------------------------------
% DGP4: 
%%------------------------------------------------------------------------- 
% 0. Generate component label
G     = 3;
omega = [0.45 0.35 0.2];
g     = mnrnd(1,omega,N);
mg    = sum(g);

%==== Export 'g' and 'omega' as extra output
extout.g     = g;
extout.omega = omega;

% 1. Generate explanatory variables
%===== for  W-Equation
c_1 = rand([N,1]); 
c1  = rand([N,N]); 
for i=1:N 
    for j=1:N 
        if c_1(i,:)>=0.7 && c_1(j,:)>=0.7 
            c1(i,j)=1; 
        elseif c_1(i,:)<=0.3 && c_1(j,:)<=0.3 
            c1(i,j)=1; 
        else 
            c1(i,j)=0; 
        end 
        if i==j 
            c1(i,j)=0; 
        end 
    end 
end 

C = [c1]; 

%===== for  Y-Equation
x = randn(N,2)*sqrt(9);
X = [x]; 

% 2. Generate coefficients
%===== for  W-Equation
gamma   = [1.5]; 

%===== for  Y-Equation
beta = [-0.5 -0.75;
            0.5  0.8;
            -1.0  1.2]';

rho = [-0.3 0.3 0.6];

beta_til    = g*beta';
rho_til  = g*rho';

% 3. Generate covariance matrix
mu      = [0.5 0]; 
sigmasq = [1.0 0.75 0.5];
kappa   = [0 0 0];  % corr.coef = 0 - exogenous network setting
kappa_til = g*kappa';

a = mu(1) + sqrt(1)*randn(N,1);              % shared latent variable, restricted to normal distribution var = 1
u = mu(2) + sqrt(g*sigmasq').*randn(N,1);    % error term in Y-Equation

% 4. W-Equation (Network Formation)
p = zeros(N,N); % probability of forming a link
w = zeros(N,N); % network link
for i=1:N 
    for j=1:N 
        psi= gamma(1)*c1(i,j) + a(i,1) + a(j,1); 
        switch options.nwdist 
            case 'logit' 
                p(i,j)=exp(psi)/(1+exp(psi));      % Logit setting 
            case 'probit' 
                p(i,j) = normcdf(psi,0,1);         % Probit setting 
        end 
        if rand(1)<=p(i,j) 
            w(i,j)=1; 
        else 
            w(i,j)=0; 
        end 
        if i==j 
            w(i,j)=0; 
        end 
    end 
end 
%===== Network Matrix
W = w;     % save 0-1 matrix 
%========== Normalize the rows 
Wmat = W; 
ridx = find(sum(Wmat, 2)~=0); 
Wmat(ridx,:) = bsxfun(@rdivide, Wmat(ridx,:), sum(Wmat(ridx,:), 2)); 
Wr = Wmat; % save row-normalised matrix 

% 5. Y-Equation (Network Interaction - SAR)
S = ( speye(N)-diag(rho_til)*Wr )^(-1);
Y = S*(sum(beta_til.*X,2) + kappa_til.*a + u); 

%==== Adjust c to obtain desirable options.R2 (or SNR)
signal_part = sum([X a]*[beta; kappa],2)'*( speye(N)-diag(rho_til)*Wr )^(-2)*sum([X a]*[beta; kappa],2);
noise_part  = (sigmasq*omega')*trace(( speye(N)-diag(rho_til)*Wr )^(-2));

c = 1; %default
if (isfield(options,'R2'))
    c = (signal_part/noise_part)*((1-options.R2)/options.R2);
    sigmasq = c*sigmasq;
    noise_part  = (sigmasq*omega')*trace(( speye(N)-diag(rho_til)*Wr )^(-2));
    u = mu(2) + sqrt(g*sigmasq').*randn(N,1);  % adjust error term in Y-Equation
    Y = S*(sum(beta_til.*X,2) + kappa_til.*a + u); 
end
%========= Check
extout.R2 = signal_part/(signal_part+noise_part);
end
end

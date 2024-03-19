function [fvalue] = logliknc_mixSAR(Y, X, Wr, lambda0, beta0, eta0, sigmav0)
% Function to evalute the log likelihood value of a mixSAR model 
%
% =====================================================================================
% Written by Duong Trinh
% University of Glasgow
% This version: June 2023
% =====================================================================================

n = size(Y,1); 

invSigmav0 = diag(1./sigmav0);

S0 = speye(n) - diag(lambda0)*Wr;

C0 = S0*Y - diag(X*beta0') - diag(Wr*X*eta0');

CC0 = (C0'*invSigmav0*C0)/2;

fvalue = log(det(S0)) - CC0;

end
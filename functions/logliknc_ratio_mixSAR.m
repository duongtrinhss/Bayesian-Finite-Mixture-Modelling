function [fvalue] = logliknc_ratio_mixSAR(Y, X, Wr, lambda0, lambda1, betag, sigmavg, points)
% Function to evalute the log likelihood ratio of a mixSAR model 
%
% =====================================================================================
% Written by Duong Trinh
% University of Glasgow
% This version: June 2023
% =====================================================================================

n = size(Y,1); 

S0 = speye(n) - diag(lambda0)*Wr;
S1 = speye(n) - diag(lambda1)*Wr;

Ytilde0 = S0*Y;
Ytilde1 = S1*Y;


invSigmavg = diag(1./sigmavg);

C0 = Ytilde0(points,:) - X(points,:)*betag;
C1 = Ytilde1(points,:) - X(points,:)*betag;

CC0 = (C0'*invSigmavg*C0)/2;
CC1 = (C1'*invSigmavg*C1)/2;

fvalue = log(det(S1)/det(S0)) - CC1 + CC0;

end
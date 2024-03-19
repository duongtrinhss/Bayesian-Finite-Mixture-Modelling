function result = truncnorm_rnd(mu,sigma2,left)
% PURPOSE: random draws from a normal truncated to (left,infinity) interval
% ------------------------------------------------------
%


std = sqrt(sigma2);
% Calculate bounds on probabilities
  lowerProb = Phi((left-mu)./std);
  
% Draw uniform from within (lowerProb,1)
  u = lowerProb+(1-lowerProb).*rand(size(mu));
% Find needed quantiles
if u>=1
    u=.9999999999;
end
  result = mu + Phiinv(u).*std;

function val=Phiinv(x)
% Computes the standard normal quantile function of the vector x, 0<x<1.

val=sqrt(2)*erfinv(2*x-1);

function y = Phi(x)
% Phi computes the standard normal distribution function value at x

y = .5*(1+erf(x/sqrt(2)));

   
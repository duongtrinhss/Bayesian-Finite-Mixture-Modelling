%This function takes draws from the truncated normal distribution
%(WITH ASSUMED TRUNCATION POINT AT ZERO). The formula is taken from Chib's 
%handbook of econometrics paper, which is taken from Ripley. 
%The inputs are mu, variance, direct, where 
	%direct = 1 imples that it is 
   %truncated to the left at 0 (positive values),
%and 
   %0 implies truncated to the right at (negative values).

%rand('seed',sum(100*clock));
function [draws] = truncnorm(mu,variance,direct);
stderrs = sqrt(variance);
uniforms = rand(length(mu),1);

c = normcdf( - mu./stderrs);
p = ( c.*(1-direct) + (1-c).*direct ).*uniforms + c.*direct;
draws = mu + stderrs.*norminv(p);
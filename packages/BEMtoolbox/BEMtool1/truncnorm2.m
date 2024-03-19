%This function generalizes the m-file truncnorm, and allows left and 
%right censoring points. The input is 
%truncnorm(mu,variance,a,b) where a and b are the upper and lower bounds
%if b = 999 then the upper bound is assumed to be infinity
%if a = -999 then the lower bound is assumed to be - infinity
   %0 implies truncated to the right at (negative values).

%rand('seed',sum(100*clock));
function [draws] = truncnorm(mu,variance,a,b);
stderrs = sqrt(variance);

if a == -999
    a_term = zeros(length(mu),1);
else
a_term = normcdf( (a-mu)./stderrs);
end;

if b == 999
    b_term = ones(length(mu),1);
else
    b_term = normcdf( (b-mu)./stderrs);
end;

uniforms = rand(length(mu),1);

p = a_term + uniforms.*(b_term - a_term);

draws = mu + stderrs*norminv(p);

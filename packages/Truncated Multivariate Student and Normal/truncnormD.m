%% truncated normal generator
% generator of a vector of length(l)=length(u)
% from the standard multivariate normal distribution,
% truncated over the region [l,u];
% =========================================================================                
%
% =========================================================================
% Written by Duong Trinh
% University of Glasgow
% This version: March 2023
% =========================================================================
function [draws] = truncnormD(mu,variance,direct,varargin)
% Inputs: , direct, where 
%   mu; variance;
%   direct = 1 imples that it is truncated to the left at 0 (positive values),
%   direct = 0 implies truncated to the right at (negative values);
%   sampler = 'tn' uses tn(.) function in Botev's toolbox
%   sampler = 'trandn' uses trandn(.) function in Botev's toolbox

if (nargin == 3)
    sampler = 'trandn';
else
    sampler = varargin{1};
end
        
stderrs    = sqrt(variance);

lb       = -mu./(stderrs);
lb(direct==0) = -Inf;
ub       = -mu./(stderrs);
ub(direct==1) = Inf;

switch sampler
    case 'trandn'
        draws_temp = trandn(lb,ub);
    case 'tn'
        draws_temp = tn(lb,ub);
end

draws      = mu + stderrs.*draws_temp;
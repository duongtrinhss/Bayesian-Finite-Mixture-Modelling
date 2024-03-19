function [perm] = aic(param,ncomp)
% AIC: artificial identifiability constraint for label-switching issue
% INPUT: param: selected parameter (size niterxncomp) to implement ordering contraint

if (size(param,2)~=ncomp)
    error('Wrong size for the selected parameter')
end

[~,perm] = sort(param,2,'descend');

end
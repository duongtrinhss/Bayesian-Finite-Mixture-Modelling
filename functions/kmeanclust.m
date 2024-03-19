function [perm] = kmeanclust(parclust,ncomp)
% algorithm to handle label-switching
% NOTE: issue with empty cluster (non-permutation)

parclust = permute(parclust,[3 1 2]); %permute such that first index is equal to group, second index to MCMC iteratio, third index to the components of group specific parameter
parclust = reshape(parclust,size(parclust,1)*size(parclust,2),size(parclust,3));
[S,clu]  = kmeans(parclust,ncomp,'EmptyAction','singleton');

perm = reshape(S,ncomp,size(S,1)/ncomp)'; % reshape nsave times ncomp

end


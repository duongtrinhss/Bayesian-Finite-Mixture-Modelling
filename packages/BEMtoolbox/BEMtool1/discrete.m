%This function takes draws from a discrete distribution with certain 
%probabilities.  The inputs are values, prob


function out = discrete(values,prob);
prob2 = cumsum(prob);
rand_num = rand(1,1);
tempa = prob2 - rand_num;
points = find(tempa>0);

out = points(1);

%if points==1

%totalmat  = [];
%for i = 1:length(values);
%   tempp = values(i,1) * ones( round( 100*prob(i,1)),1)';
%   totalmat = [totalmat tempp];
%end;

%w = length(totalmat);
%h = unidrnd(w);
%out = totalmat(1,h);
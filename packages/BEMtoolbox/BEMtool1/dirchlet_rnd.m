
%This function obtains draws from the Dirchlet distribution. 
%In this version, the ouput is a single vector. 
%syntax: out = dirchlet_rnd(theta), where theta is the x times 1 vector 
    %of parameters. 
    
function out =dirchlet_rnd(theta);
k = length(theta);
a = zeros(k,1);
%draw first part from beta;
a(1) = beta_rnd(1,theta(1),sum(theta(2:k)));

%draw next ones.
for j = 2:k-1;
    phi(j) = beta_rnd(1,theta(j),sum(theta(j+1:k)));
    a(j) = (1-sum(a(1:j-1)))*phi(j);
end;

a(k) = 1 - sum( a(1:k-1));
out = a;
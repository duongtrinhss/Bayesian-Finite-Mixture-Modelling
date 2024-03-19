
%This function obtains one draw from the multinomial distribution. 
%with probability vector P. 
%
%the syntax is function out = multinom_rnd(P);
%the method os taken from Gelman, Stern and Rubin's appendix.

function out = multinom_rnd(P);
[n,k] = size(P);
a = zeros(n,k);
a(:,1) = binornd(1,P(:,1));

for j = 2:k-1;
   if j ==2;
       a(:,j) = binornd(1 - a(:,1), ( P(:,2)./sum(P(:,j:k)')' ) );
   elseif j~=2 
     a(:,j) = binornd(1 - sum(a(:,1:j-1)')', ( P(:,j)./sum(P(:,j:k)')' ) );
    end;    
 end;
 
 a(:,k) = 1 - sum( a(:,1:k-1)' )';
 
 out = a;
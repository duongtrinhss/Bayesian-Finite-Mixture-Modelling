%THIS FUNCTION uses Nobile's algorithm to draw from the 
%wishart density, fixing the first element equal to ONE.
%The syntax is nobile_wishart(v,S). The draws are obtained such that 
%The expected value of the draw from this 
%UNCONDITIONAL WWishart distribution is vS^-1.
function out = wishrand(v,S);
randn('seed',sum(100*clock));

p = length(S);	R = chol(inv(S));	L = R';	%Now, W*W' = S^-1, as needed

chivect = zeros(p,1);

for j = 1:p;
   chivect(j,1) = sqrt(chi2rnd(v+1-j));		%Assemble chisquared draws for diagonal elements.
end;

mattemp = randn(p,p);

matuse = zeros(p,p);					%Mattemp is a matrix of the chisquares on the diagonal
												%and the remaining elements standard normal
for j = 1:p;
  % if j==1;
  %     matuse(j,:) = [sqrt(1)/L(1,1) mattemp(j,2:p)];
     if j==1;
      matuse(j,:) = [chivect(j,1) mattemp(j,2:p)];
  elseif j==p;
      matuse(j,:) = [mattemp(j,1:p-1) inv(L(p,p)*sqrt(1))];
      %elseif j==p;																		%Take the first and last 
     % matuse(j,:) = [mattemp(j,1:p-1) chivect(j,1)];	%rows seperately
   else	
      matuse(j,:) = [mattemp(j,1:j-1) chivect(j,1) mattemp(j,j+1:p)];
   end;
end;
%chivect
%matuse
lowertrian = tril(matuse);
out = L*lowertrian*lowertrian'*L';

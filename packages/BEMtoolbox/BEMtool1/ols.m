%This function computes some standard ols output.  The inputs 
%are [bhat stderr tstat sigsquare rsquare] = ols(x,y)
%Where x is an nxk design martix, and y (nx1) is the 
%dependent variable.  bhat (kx1) is the coefficient estimates, 
%stderr (kx1) is standard errors, tstat gives the t-statistics
%sigsquare (s^2) is the estimated residual variance, and 
%rsquare is the coefficient of determination.
 
function [bhat, stderr,tstat,ssquare,rsquare] = ols(x,y);
bhat = inv(x'*x)*(x'*y);
dim1 = size(y,1);
dim2 = size(x,2);
const = 1/(dim1);
ssquare = const*(y-x*bhat)'*(y-x*bhat);
varcov = ssquare*inv(x'*x);

for i = 1:dim2;
	stderr(i,1) = sqrt(varcov(i,i));
end;

tstat = bhat./stderr;
yhat = x*bhat;
ybar = mean(y);
rsquare = ( sum( (yhat - ybar).^2) )/ sum( (y - ybar).^2 );


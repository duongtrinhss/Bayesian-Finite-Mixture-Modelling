function [fvalue] = betacenlpdf(x,d)
%BETACENLPDF Logarithm of density function of beta distribution centered on
%a mean value of zero
fvalue = (d-1)*log(1+x)+(d-1)*log(1-x)-(2d-1)*log(2)-betaln(d,d);
end

%Test
% x = -1:0.1:1;
% d = 1.01;
% f = betacenlpdf(x,d);
% exp(f)
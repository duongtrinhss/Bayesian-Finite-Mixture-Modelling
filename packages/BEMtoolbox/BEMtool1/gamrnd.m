function r = gamrnd(a,b,m,n);
%GAMRND Random matrices from gamma distribution.
%   R = GAMRND(A,B) returns a matrix of random numbers chosen   
%   from the gamma distribution with parameters A and B.
%   The size of R is the common size of A and B if both are matrices.
%   If either parameter is a scalar, the size of R is the size of the other
%   parameter. Alternatively, R = GAMRND(A,B,M,N) returns an M by N matrix. 
%
%   Some references refer to the gamma distribution
%   with a single parameter. This corresponds to GAMRND
%   with B = 1. (See Devroye, pages 401-402.)

%   GAMRND uses a rejection or an inversion method depending on the
%   value of A. 

%   References:
%      [1]  L. Devroye, "Non-Uniform Random Variate Generation", 
%      Springer-Verlag, 1986

%   B.A. Jones 2-1-93
%   Copyright (c) 1993-96 by The MathWorks, Inc.
%   $Revision: 2.3 $  $Date: 1996/02/13 23:11:00 $

if nargin < 2, 
    error('Requires at least two input arguments.'); 
end


if nargin == 2
    [errorcode rows columns] = rndcheck(2,2,a,b);
end

if nargin == 3
    [errorcode rows columns] = rndcheck(3,2,a,b,m);
end

if nargin == 4
    [errorcode rows columns] = rndcheck(4,2,a,b,m,n);
end

if errorcode > 0
    error('Size information is inconsistent.');
end

% Initialize R to zero.
r = zeros(rows,columns);

scalara = (prod(size(a)) == 1);
if scalara 
    a = a(ones(rows,1),ones(columns,1));
end

scalarb = (prod(size(b)) == 1);
if scalarb 
    b = b(ones(rows,1),ones(columns,1));
end

% If a == 1, then gamma is exponential. (Devroye, page 405).
k = find(a == 1);
if any(k)
        r(k) = -b(k) .* log(rand(size(k)));
end 

c = zeros(rows,columns);
d = zeros(rows,columns);

k = find(a < 1 & a > 0);
% (Devroye, page 418 Johnk's generator)
if any(k)
    c(k) = 1 ./ a(k);
    d(k) = 1 ./ (1 - a(k));
    accept = k;
    while(length(accept)>0),
  u = rand(size(accept));
        v = rand(size(accept));
        x = u .^ c(accept);
        y = v .^ d(accept);
        k1 = find(x + y <= 1); 
        e = -log(rand(size(k1))); 
        r(accept(k1)) = e .* x(k1) ./ (x(k1) + y(k1));
        accept(k1) = [];
    end
    r(k) = r(k) .* b(k);
end

% Use a rejection method for A > 1.
k = find(a > 1);
% (Devroye, page 410 Best's algorithm)
bb = zeros(max(k),1);
c  = bb;
if any(k)
    bb(k) = a(k) - 1;
    c(k) = 3 * a(k) - 3/4;
    accept = k;
    while(length(accept)>0);
      [m, n] = size(c(accept));
        u = rand(m,n);
        v = rand(m,n);
        w = u .* (1 - u);
        y = sqrt(c(accept) ./ w) .* (u - 0.5);
        x = bb(accept) + y;
        z = 64 * w .^ 3 .* v .^ 2;
        k1 = find(x >= 0);
        rhs = 2 * (bb(k(k1)) .* log(x(k1) ./ bb(k(k1))) - y(k1));
        k2 = find(z(k1) <= 1 - 2 * y(k1) .^2 ./ x(k1));
        r(accept(k1(k2))) = x(k1(k2)); 
        k3 = find(log(z(k1)) <= rhs & z(k1) > 1 - 2 * y(k1) .^2 ./ x(k1));
        r(accept(k1(k3))) = x(k1(k3));
        k1 = k1(:);
        omit = [k1(k2(:));k1(k3(:))];
        accept(omit) = [];
    end
        r(k) = r(k) .* b(k);
end

% Return NaN if b is not positive.
if any(any(b <= 0));
    if prod(size(b) == 1)
        tmp = NaN;
        r = tmp(ones(rows,columns));
    else
        k = find(b <= 0);
        tmp = NaN;
        r(k) = tmp(ones(size(k)));
    end
end

% Return NaN if a is not positive.
if any(any(a <= 0));
    if prod(size(a) == 1)
        tmp = NaN;
        r = tmp(ones(rows,columns));
    else
        k = find(a <= 0);
        tmp = NaN;
        r(k) = tmp(ones(size(k)));
    end
end

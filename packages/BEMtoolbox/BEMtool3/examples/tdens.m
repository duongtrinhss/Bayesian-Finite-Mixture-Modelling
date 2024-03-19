function dens = tdens(apoint,amean,acapv,adof)
%evaluate univariate t density with arguments amean, acapv and adof 
%at apoint

lintcon=.5*log(acapv)-.5*adof*log(adof)+.5*log(pi)+gammaln(.5*adof)...
    -gammaln(.5*(adof+1));
dens = -.5*(adof+1)*log(adof+((apoint-amean)^2)/acapv)-lintcon;
dens=exp(dens);






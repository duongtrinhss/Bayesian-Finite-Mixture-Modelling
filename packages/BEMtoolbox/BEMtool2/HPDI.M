function bounds = hpdi(adraw,hperc)
%This function takes a vector of MCMC draws and calculates
%a hperc-percent HPDI
ndraw=size(adraw,1);
botperc=round(hperc*ndraw);
topperc=round((1-hperc)*ndraw);
temp = sort(adraw,1);
bounds=[temp(topperc,1) temp(botperc,1)];


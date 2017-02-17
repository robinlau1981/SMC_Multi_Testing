function Res=ilogit(x)
N=length(x);
Res=ones(N,1)./(ones(N,1)+exp(-x));
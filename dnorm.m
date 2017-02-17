function Res=dnorm(z, mu0, sig0)
N=length(z);
Res=zeros(N,1);
for i=1:N
    Res(i)=normpdf(z(i),mu0,sig0(i));
end
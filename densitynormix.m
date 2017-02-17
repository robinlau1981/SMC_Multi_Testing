function density = densitynormix(y, sigma2, weights, mu,tau2)
% return mixture density and density values of each mixture component
% here y is assumed to be a single data item
ncomps = length(mu);
normalized_weights = weights/sum(weights);
density.mix=0; % mixture density value at y
density.comps=zeros(1,ncomps);  % density values w.r.t. mixture components at y
% thisdens=zeros(ncases,1);
% mysd=zeros(ncomps,1);
% z=zeros(ncases,1);
for j=1:ncomps
    mysd = sqrt(sigma2 + tau2(j));
    z = (y-mu(j))./mysd;
    density.comps(j)=normpdf(z)./mysd;
    thisdens = normalized_weights(j)*density.comps(j);
    density.mix = density.mix +thisdens;
end
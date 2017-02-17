function density = marnormix(y, sigma2, weights, mu,tau2)
% NumericVector marnormix(NumericVector y, NumericVector sigma2, NumericVector weights,  NumericVector mu, NumericVector tau2)
% y and sigma2 are n-vectors of observations (y[i]) and observation-level variances (sigma2[i])
% weights, mu, and tau2 are p-vectors of weights, means, and variances in a K-component mixture model
% returns the marginal/predictive density of y[i] under a mixture of normals prior and Gaussian observation model
ncases = length(y);
ncomps = length(mu);
normalized_weights = weights/sum(weights);
density=zeros(ncases,1);
% thisdens=zeros(ncases,1);
% mysd=zeros(ncomps,1);
% z=zeros(ncases,1);
for j=1:ncomps
    mysd = sqrt(sigma2 + tau2(j));
    z = (y-mu(j))./mysd;
    thisdens = normalized_weights(j)*(normpdf(z)./mysd);
    for i=1:ncases
        density(i) = density(i)+thisdens(i);
    end
end



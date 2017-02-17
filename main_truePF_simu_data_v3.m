% 2017-01-31 Particle filter for sequential test without MCMC based
% training
% is successful
% Ref: A One-Pass Sequential Monte Carlo Method for Bayesian Analysis of Massive Datasets
% 2016-10-30
% Ref: Bayesian FDRreg R package from Github
clear;
close all;

% Simulated data
P = 2;
N = 1e3;
%Ni=floor(1*N/2); % number of data for initial training
% betatrue=[-3.5 repmat(1/sqrt(P),1,P)];
% X=randn(N,P);
load Res_bfdrpf_simu_data;
clear bfdr1 myfindings guess;
psi=[ones(N,1) X]*betatrue';
wsuccess=1./(1+exp(-psi));

% Some theta's are signals, most are noise
gammatrue=binornd(ones(N,1),wsuccess,N,1);
disp('number of 0s in gammatrue');
sum(gammatrue==0)
disp('number of 1s in gammatrue');
sum(gammatrue==1)

% Density of signals
thetatrue = normrnd(3,0.5,N,1);
ind=find(gammatrue==0);
thetatrue(ind)=zeros(length(ind),1);
z=normrnd(thetatrue,1);



%% Preparation for PF
empiricalnull=0;
ps=1e4; % particle size for importance sampling for estimate beta
std_new_comp=1;
Ni=0;
Prob_alt=.1;
Prob_null=1-Prob_alt;
N_alt=1;%Ni*Prob_alt; % particle size of alternative distribution
N_null=9;%Ni*Prob_null;  % particle size of null distribution
Q2=.5;
beta.mean=betatrue;%[-3 1/ 1];%mean(bfdr1.coefficients(1:10:end,:),1);
beta.cov=diag([10 10 10]);%cov(bfdr1.coefficients(1:10:end,:));
betapar=mvnrnd(beta.mean,beta.cov,ps);
for i=1:ps
    M0(i).mu=0;%bfdr1.mu0; % M0 : noise model
    M0(i).sig=1.5;%mean(bfdr1.sig0(1:10:end));
    M1(i).weights=1;%[1/3 1/3 1/3];%mean(bfdr1.weights(1:10:end,:),1);
    M1(i).means=3;%[1 3 5];%mean(bfdr1.means(1:10:end,:),1);
    M1(i).variance=20;%[5 5 5];%mean(bfdr1.vars(1:10:end,:),1);
    M1(i).ncomps=1;%3;%bfdr1.ncomps;
    N_alt(i)=1;%Ni*Prob_alt; % particle size of alternative distribution
    N_null(i)=9;%Ni*Prob_null;  % particle size of null distribution
end
b=(4/((P+1+2)*ps))^(1/(P+1+4)); % kernel width Ref:Eqn. 10 in A One-Pass Sequential Monte Carlo Method for Bayesian Analysis of Massive Datasets
a=sqrt(1-b^2);
tic;
for t=1:N-Ni
    t/N
    lik=ones(ps,1);
    for i=1:ps
        f0(i) = dnorm(z(Ni+t), M0(i).mu, M0(i).sig);
        f1(i) = densitynormix(z(Ni+t), M0(i).sig.^2, M1(i).weights, M1(i).means, M1(i).variance);        
        Psi =cbind(1,X(Ni+t,:))*betapar(i,:)';
        W = ilogit(Psi);
        lik(i) = W*f1(i).mix+(1-W)*f0(i);
    end
    lik=lik/sum(lik);
    ess(t)=1/sum(lik.^2)/ps;
    [~,ind]=max(lik);
    beta_est(t,:)=betapar(ind,:);
    M0_est=M0(ind);
    M1_est=M1(ind);
    % Resampling
    outIndex = residualR(1:ps,lik);
    betapar = betapar(outIndex,:);
    M0=M0(outIndex);
    M1=M1(outIndex);    
    N_alt=N_alt(outIndex);
    N_null=N_null(outIndex);
    f0=f0(outIndex);
    f1=f1(outIndex);
    % Particle Moving of beta
    mean_particle=mean(betapar,1);
    error=betapar-repmat(mean_particle,ps,1);
    cov_particle=error'*error/ps;
    betapar_tmp=a*betapar+repmat((1-a)*mean_particle,ps,1);  % Ref:Fig.3 in A One-Pass Sequential Monte Carlo Method for Bayesian Analysis of Massive Datasets
    betapar=betapar_tmp+mvnrnd(zeros(1,P+1),b^2*cov_particle,ps);
    % Particle Moving - mixture parameters
    for i=1:ps
        %% online k-means
        %% Ref: Adaptive background mixture models for real-time tracking       
        Psi2 =cbind(1,X(Ni+t,:))*betapar(i,:)';
        W2 = ilogit(Psi2);
        Pp = W2*f1(i).mix/(W2*f1(i).mix+(1-W2)*f0(i)); % Posterior probability of non-Null
        if Pp<1-Q2 % z(Ni+t) belongs to null
            rou=1/(1+N_null(i));
            if empiricalnull==1
                M0(i).mu=(1-rou)*M0(i).mu+rou*z(Ni+t);
            else
                M0(i).mu=0;
            end
            M0(i).sig=sqrt((1-rou)*M0(i).sig^2+rou*(z(Ni+t)-M0(i).mu)^2);
            N_null(i)=N_null(i)+1;
        else
            alpha=1/(1+N_alt(i)); % close to 0
            M1(i).weights=(1-alpha)*M1(i).weights;
            dis=zeros(1,M1(i).ncomps);
            for j=1:M1(i).ncomps
                dis(j)=abs(z(Ni+t)-M1(i).means(j))/sqrt(M1(i).variance(j));
            end
            [min_dis,ind]=min(dis);
            if min_dis>2.5 && M1(i).ncomps<3  % z(Ni) does not belong to any component
                M1(i).ncomps=M1(i).ncomps+1;
                M1(i).means(M1(i).ncomps)=z(Ni+t);
                M1(i).weights(M1(i).ncomps)=alpha;
                M1(i).variance(M1(i).ncomps)=std_new_comp^2;
            else
                rou=alpha/(M1(i).weights(ind)+alpha);
                M1(i).means(ind)=(1-rou)*M1(i).means(ind)+rou*z(Ni+t);
                M1(i).weights(ind)=M1(i).weights(ind)+alpha;
                M1(i).variance(ind)=(1-rou)*M1(i).variance(ind)+rou*(z(Ni+t)-M1(i).means(ind))^2;
            end
            N_alt(i)=N_alt(i)+1;
        end        
    end
    if  t==1
        betapar_1st_iter=betapar;
    elseif t==(N-Ni)/2 
        betapar_half_iter=betapar;        
    end
%     if mod(t,100)==0
%         idx_time=t/100;
%         time_elapsed(idx_time)=toc;
%         tic;
%     end
end
% time_elapsed(end+1)=toc;

%% Bayes test
Psi =cbind(1,X)*beta_est(end,:)';
W = ilogit(Psi);
f0_full = dnorm(z, M0_est.mu, M0_est.sig*ones(N,1));
f1_full = marnormix(z, M0_est.sig^2, M1_est.weights, M1_est.means, M1_est.variance);
PostProb = W.*f1_full./((1-W).*f0_full + W.*f1_full);
Res.BF=(W.*f1_full)./((1-W).*f0_full);
guess=zeros(N,1);
ind =find(Res.BF>(1-Q2)/Q2);
guess(ind)=ones(length(ind),1);
% % Extract findings at level FDR = Q
% myfindings = find(bfdr1.FDR <= Q);
% hist(z(myfindings),50);
% % table(truth = gammatrue, guess = {bfdr1$FDR <= Q})
% guess = bfdr1.FDR <= Q;
table(1,1)=length(find(gammatrue==0 & guess==0 ));
table(1,2)=length(find(gammatrue==0 & guess==1 ));
table(2,1)=length(find(gammatrue==1 & guess==0 ));
table(2,2)=length(find(gammatrue==1 & guess==1 ));
% % table(1,1)=length(intersect(find(gammatrue==0),find(guess==0)));
% % table(1,2)=length(intersect(find(gammatrue==0),find(guess==1)));
% % table(2,1)=length(intersect(find(gammatrue==1),find(guess==0)));
% % table(2,2)=length(intersect(find(gammatrue==1),find(guess==1)));
correct_findings=find(gammatrue==1 & guess==1 );
wrong_findings=find(gammatrue==0 & guess==1 );
table
%save Res_truepf_v3_simu_data gammatrue thetatrue wsuccess X z Ni N table gammatrue guess empiricalnull M0_est M1_est Q Q2 betatrue beta_est correct_findings wrong_findings betapar ps ess betapar_1st_iter betapar_half_iter;
figure,
histogram(betapar(:,1), 20,'Normalization','pdf');hold on;
plot([betatrue(1) betatrue(1)],[0 5.5],'r','LineWidth',4);hold off;axis([-4.2 -2.4 0 5.5]);grid on
figure,
histogram(betapar_1st_iter(:,1), 20,'Normalization','pdf');hold on;
plot([betatrue(1) betatrue(1)],[0 3.3],'r','LineWidth',4);hold off;axis([-4.2 -2.4 0 3.3]);grid on
figure,
histogram(betapar_half_iter(:,1), 20,'Normalization','pdf');hold on;
plot([betatrue(1) betatrue(1)],[0 5.5],'r','LineWidth',4);hold off;axis([-4.2 -2.4 0 5.5]);grid on

figure,plot(1:t,beta_est(:,1),1:t,betatrue(1)*ones(1,t),'r','LineWidth',2);grid on;
xlabel('t');ylabel('\beta_0');
figure,plot(1:t,beta_est(:,2),1:t,betatrue(2)*ones(1,t),'r','LineWidth',2);grid on;
xlabel('t');ylabel('\beta_1');
figure,plot(1:t,beta_est(:,3),1:t,betatrue(3)*ones(1,t),'r','LineWidth',2);grid on;
xlabel('t');ylabel('\beta_2');
figure,plot(1:t,ess);xlabel('t');ylabel('NESS');axis([0 10000 0 1.1]);grid on;
function [alpha, mu, sigma, likelihood_vals] = em_gmm(x,K)
%% x: N*d
%% k:number of gaussian component
%% implement as the "The EM Algorithm for Gaussian Mixtures, Probabilistic Learning: Theory and Algorithms, CS 274A", 
%% http://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
%% no optimization

[N,d]=size(x);

%% initilize membering weights
w=rand(N,K);
w=bsxfun(@times, w, 1./(max(sum(w, 2), eps)));%row sum to 1

%% compute weights of mixture
alpha=sum(w,1)./N;

%% get initialized mu
% mu=zeros(K,d);
% Nk=sum(w,1);
% for k=1:K
%     for i=1:N
%         mu(k,1:d)=mu(k,1:d)+w(i,k)*x(i,1:d);
%     end
%     mu(k,1:d)=mu(k,1:d)./Nk(k);
% end
% % The following function is equivalant to the code commented above.
mu = update_mu(x,w,K);

%% get initialized sigma
% sigma=zeros(d,d,K);
% for k=1:K
%     for i=1:N
%         sigma(:,:,k)=sigma(:,:,k)+w(i,k)*(x(i,1:d)-mu(k,1:d))'*(x(i,1:d)-mu(k,1:d));
%     end
%     sigma(:,:,k)= sigma(:,:,k)./Nk(k);
% end
% % The following function is equivalant to the code commented above.
sigma = update_sigma(x,w,mu,K);

iterMax=100;
likelihood_vals=zeros(1,iterMax);

%% computer initial likelihood
likelihood_vals(1)=likelihood(x,alpha,mu,sigma);
fprintf('%3d -- > %.6f\n', 1, likelihood_vals(1));
figure,plot(1,likelihood_vals(1),'r.'),drawnow, hold on

for iter=2:iterMax
    
    %% update w
    %     w_new=zeros(N,K);
    %     for i=1:N
    %         for k=1:K
    %             w_new(i,k)=alpha(k)*gauss_val(x(i,1:d),mu(k,1:d),sigma(:,:,k));
    %         end
    %     end
    %     w_new=bsxfun(@times, w_new, 1./(max(sum(w_new, 2), eps)));%row sum to 1
    %     w=w_new;
    % % The following function is equivalant to the code commented above.
    w = update_w(x,alpha,mu,sigma,K);
    
    alpha=sum(w,1)./N;
    
    %% update mu
    %     mu_new=zeros(K,d);
    %     Nk=sum(w,1);
    %     for k=1:K
    %         for i=1:N
    %             mu_new(k,1:d)=mu_new(k,1:d)+w(i,k)*x(i,1:d);
    %         end
    %         mu_new(k,1:d)=mu_new(k,1:d)./Nk(k);
    %     end
    %     mu=mu_new;
    % % The following function is equivalant to the code commented above.
    mu = update_mu(x,w,K);
    
    %% update sigma
    %     sigma_new=zeros(d,d,K);
    %     for k=1:K
    %         for i=1:N
    %             sigma_new(:,:,k) = sigma_new(:,:,k)+w(i,k)*(x(i,1:d)-mu(k,1:d))'*(x(i,1:d)-mu(k,1:d));
    %         end
    %         sigma_new(:,:,k) = sigma_new(:,:,k)./Nk(k);
    %     end
    %     sigma=sigma_new;
    % % The following function is equivalant to the code commented above.
    sigma = update_sigma(x,w,mu,K);
    
    %% compute new likelihood
    likelihood_vals(iter)=likelihood(x,alpha,mu,sigma);
    fprintf('%3d -- > %.6f\n', iter, likelihood_vals(iter));
    plot(iter,likelihood_vals(iter),'r.'),drawnow, hold on
end
title('likelihood'),drawnow, hold off


function val=gauss_val(x,mu,sigma)
%% multivariate gaussian density
d=size(x,2);
val=exp(-0.5*(x-mu)*inv(sigma+0.0001*eye(d,d))*(x-mu)')/((2*pi)^(d/2)*((det(sigma+0.0001*eye(d,d)))^(0.5)));

function val=likelihood(x,alpha, mu,sigma)
%% log-likelihood function
[N,d]=size(x);
K=size(mu,1);
val=0;
for i=1:N
    temp=0;
    for k=1:K
        temp=temp+alpha(k)*gauss_val(x(i,1:d),mu(k,1:d),sigma(:,:,k));
    end
    if temp>0
        val=val+log(temp+eps);
    end
end

function w_new = update_w(x,alpha,mu,sigma,K)
%% update membering weights
[N,d]=size(x);
w_new=zeros(N,K);
for i=1:N
    for k=1:K
        w_new(i,k)=alpha(k)*gauss_val(x(i,1:d),mu(k,1:d),sigma(:,:,k));
    end
end
w_new=bsxfun(@times, w_new, 1./(max(sum(w_new, 2), eps)));%row sum to 1

function mu_new = update_mu(x,w,K)
%% update mu
[N,d]=size(x);
mu_new=zeros(K,d);
Nk=sum(w,1);
for k=1:K
    for i=1:N
        mu_new(k,1:d)=mu_new(k,1:d)+w(i,k)*x(i,1:d);
    end
    mu_new(k,1:d)=mu_new(k,1:d)./Nk(k);
end

function sigma_new = update_sigma(x,w,mu,K)
%% update sigma
[N,d]=size(x);
Nk=sum(w,1);
sigma_new=zeros(d,d,K);
for k=1:K
    for i=1:N
        sigma_new(:,:,k) = sigma_new(:,:,k)+w(i,k)*(x(i,1:d)-mu(k,1:d))'*(x(i,1:d)-mu(k,1:d));
    end
    sigma_new(:,:,k) = sigma_new(:,:,k)./Nk(k);
end

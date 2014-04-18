
x=[];
load x;
K=3;
[alpha, mu, sigma, likelihood_vals] = em_gmm(x,K);
figure,
plot(x(:,1),x(:,2),'b.')
hold on
plot(mu(:,1),mu(:,2),'ro')
hold on
x1 = -4:.2:8; x2 = -8:.2:1;
[X1,X2] = meshgrid(x1,x2);
for k=1:K
    F = mvnpdf([X1(:) X2(:)],mu(k,:),sigma(:,:,k));
    F = reshape(F,length(x2),length(x1));
    contour(x1,x2,F,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999]);
    hold on
end











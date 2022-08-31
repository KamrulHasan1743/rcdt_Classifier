function y = RCDT_features(I)

I_domain = [0, 1];
Ihat_domain = [0, 1];
theta_seq = 0:4:179;
rm_edge = 1;
I_hat = RCDT(I_domain, I, Ihat_domain, theta_seq, rm_edge);
y=I_hat(:)';
end


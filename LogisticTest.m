function [y_prob, y_hat] = LogisticTest(LR,X)

w = LR.w;
c = LR.c;

y_prob = 1./ (1+ exp(-(X*w+c) ) );
y_hat = y_prob;
y_hat(y_hat>=0.5) = 1;
y_hat(y_hat<0.5) = -1;

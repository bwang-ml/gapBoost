function LR = weightLogisticTrain(X,y,gamma,opts)
lambda = 0.0000001;

if opts.sampling == 1
    [X, y] = WeightedSampling(X,y,gamma);
else
    opts.sWeight = gamma;
end

[w, c, ~, ~] = LogisticR(X, y, lambda, opts);

LR.w = w;
LR.c = c;
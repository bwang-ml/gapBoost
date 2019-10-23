clear
clc
close all
load('20Newsgroups.mat');


Taskind = [11 19 10 18; 4 14 5 12; 8 13 9 15; 15 17 14 19; 4 8 5 9; 2 20 3 19];             % task IDs
ratios = [0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8];                                  % ratio of target training sample
numRatio = length(ratios);                              
numRandom = 20;                                                                             % repeat 20 times
numRho = 9;                                                                                 % vary the value of rho_S



K = size(Taskind,1);
TotalErr = zeros(K,numRho,numRatio,numRandom);


dims = 100;                                                                                 % we reduce the dimensionality to 100 by PCA


params.rsL2 = 0;                                                                       % regularization parameter for logistic regression
params.eta = 0;
params.M = 20;                                                                              % number of base learners
params.rhoT = 1;                                                                            % we simply set rho_T = 1
params.sampling = 1;                                                                        % we sample the examples according to the distribution at each boosting iteration




for i = 1:K     % loop over tasks

    idxSub = Taskind(i,:);
    idxS0 = find(gnd == idxSub(1));
    idxS1 = find(gnd == idxSub(2));
    idxT0 = find(gnd == idxSub(3));
    idxT1 = find(gnd == idxSub(4));

    for nRho = 1:numRho        % loop over values of rho_S
        params.rhoS = 0.1*nRho;

        for j = 1:numRatio        % loop over ratios of target traning sample

            % prepare the data
            idx0 = idxS0;
            idx1 = idxS1;
            XS0 = fea(idx0,:);
            XS1 = fea(idx1,:);
            yS0 = -ones(length(idx0),1);
            yS1 = ones(length(idx1),1);
            XS = [XS0;XS1];
            yS = [yS0;yS1];

            idx0 = idxT0;
            idx1 = idxT1;
            XT0 = fea(idx0,:);
            XT1 = fea(idx1,:);
            yT0 = -ones(length(idx0),1);
            yT1 = ones(length(idx1),1);
            XT = [XT0;XT1];
            yT = [yT0;yT1];

            XS(XS>1) = 1;
            XT(XT>1) = 1;

            [X,S,~] = svds([XS;XT],dims);

            for k = 1:numRandom     

                numS = length(yS);
                numT = length(yT);

                XS = X(1:numS,:);
                XT = X(numS+1:end,:);
                yT = [yT0;yT1];

                % randomely split the target data
                ind = randperm(numT);
                XT = XT(ind,:);
                yT = yT(ind);
                ratio = ratios(j);
                n = round(numT*ratio);
                XTtrain = XT(1:n,:);
                XTtest = XT(n+1:end,:);
                yTtrain = yT(1:n);
                yTtest = yT(n+1:end);

                ensembles = gapBoostTrain(XTtrain,yTtrain,XS,yS,params);
                [f, err] = gapBoostTest(ensembles,XTtest,yTtest,params);
                TotalErr(i,nRho,j,k) = err;

            end    
        end
    end
end


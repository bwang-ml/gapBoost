function ensembles = gapBoostTrain(XT,yT,XS,yS,params)


M = params.M;
rsL2 = params.rsL2;
eta = params.eta;
rhoS = params.rhoS;
rhoT = params.rhoT;

numT = length(yT);
numS = length(yS);
num = numT + numS;


gammaT = ones(numT,1)/numT;
gammaS = ones(numS,1)/numS;
gamma = ones(num,1)/num;

gMax = sqrt(1/numT);
% gMax = 1;

% parameters for logistic regression.
% dependency: SLEP_package_4.1 toolbox 

opts=[];
% Starting point
opts.init=2;        % starting from a zero point
% Termination 
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=100;    % maximum number of iterations
% Normalization
opts.nFlag=0;       % without normalization
% Regularization
opts.rFlag=0;       % the input parameter 'rho' is a ratio in (0, 1)
opts.mFlag=1;       % smooth reformulation 
opts.lFlag=1;       % adaptive line search
opts.tFlag=2; 
opts.sampling = params.sampling;

optsT = opts;
optsS = opts;


optsT.rsL2 = rsL2*eta;
optsS.rsL2 = rsL2*eta;
opts.rsL2 = rsL2;




ensembles = cell(1,M);




X = [XS;XT];
y = [yS;yT];

for i = 1:M
    ensembles{i}.D = gamma;
    
    % train a base learner 
    ensembles{i} = weightLogisticTrain(X,y,gamma,opts);
    [~, h] = LogisticTest(ensembles{i},X);
    errs = gamma.*(h~=y);
    err = sum(errs);
    beta = (1-err)/err;
    gamma(h ~= y) = gamma(h ~= y)*beta;
  
    
   % train auxiliary learners  
    hS = weightLogisticTrain(XS,yS,gammaS,optsT);
    hT = weightLogisticTrain(XT,yT,gammaT,optsT);

    % cross test
    [~, ySS] = LogisticTest(hS,XS);
    [~, yST] = LogisticTest(hS,XT);
    [~, yTT] = LogisticTest(hT,XT);
    [~, yTS] = LogisticTest(hT,XS);

    
    for j = 1:num
        if j <= numS
            if ySS(j) ~= yTS(j)
                gamma(j) = gamma(j)*rhoS;
            end
        else
            if yTT(j-numS) ~= yST(j-numS)
                gamma(j) = gamma(j)*rhoT;
            end
        end
    end
    
        
    % weight normalization
    gamma = gamma/sum(gamma);
    gamma(gamma>gMax) = gMax;
    gamma = gamma/sum(gamma);
    
    gammaS = gamma(1:numS);
    gammaT = gamma(numS+1:end);
    gammaS = gammaS/sum(gammaS);
    gammaT = gammaT/sum(gammaT);
    
    ensembles{i}.err = err;

end
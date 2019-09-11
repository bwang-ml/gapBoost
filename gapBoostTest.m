function [f, err] = gapBoostTest(ensembles,X,y,params)

num = size(X,1);
f = nan(num,1);
y0 = zeros(num,1);
y1 = zeros(num,1);


M = params.M;
for i = 1:M
    [~, h] = LogisticTest(ensembles{i},X);
    err = ensembles{i}.err;
    alpha = log((1-err)/err);

    y0(h==-1) = y0(h==-1)+alpha;
    y1(h==1) = y1(h==1)+alpha;
end

f(y0>y1) = -1;
f(y0<=y1) = 1;

err =  sum(f~=y)/num;
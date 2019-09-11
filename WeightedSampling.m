function [Xsam, ysam] = WeightedSampling(X,y,w,N)


[num, dim] = size(X);


if nargin == 4
    num = N;
end


Xsam = zeros(num,dim);
ysam = zeros(num,1);

cumw = cumsum(w);
wmax = cumw(end);

t = 1;
while t 
    for i = 1:num
        a = rand*wmax;
        idx = find(cumw<=a);

        if isempty(idx)
            idx = 1;
        else
            idx = idx(end)+1;
        end

        Xsam(i,:) = X(idx,:);
        ysam(i) = y(idx);

    end
    if length(unique(ysam)) == 2
        t = 0;
    end
end

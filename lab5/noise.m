function [res] = noise(p, n)
    sz = size(p);
    limit = sz(1);
    temp = p';
    
    for i = 1:n
       pos = randi([1 limit],1,1);
       temp(pos) = temp(pos) * (-1);
    end
    res = temp';
end
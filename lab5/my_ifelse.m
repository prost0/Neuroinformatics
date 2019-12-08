function [x] = my_ifelse(x)
    if x >= 0
        x = 1;
    else
        x = -1;
    end
end
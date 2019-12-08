function [args, values, target] = p1_data()
    args = 0:0.025:1;
    values = sin(4 * pi * args);
    target = ones(1,length(args));
end
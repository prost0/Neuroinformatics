function [args, values, target] = p2_data()
    args =  1.49:0.025:3.52;
    values = 1.5 * sin(power(args,2) - 6 * args + 3) - 0.8;
    target = -1 * ones(1, length(args));
end
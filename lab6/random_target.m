function P = random_target(n)
    P = randi([0 1], 1, n) * 2 - 1;
end
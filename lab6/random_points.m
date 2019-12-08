function P = random_points(n)
    rnd = makedist('Uniform',0,1.5);
    P = random(rnd,2,n);
end
function P = random_points2(n)
    rnd = makedist('Uniform',-1.5,1.5);
    P = random(rnd,2,n);
end
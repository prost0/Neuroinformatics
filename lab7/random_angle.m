function numbers = random_angle(n)
    rnd = makedist('Uniform',-pi/4, 7 * pi /4);
    numbers = random(rnd,1,n);
end
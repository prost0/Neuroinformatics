function [p] = rectanglos(n, d1, d2, alpha, x0, y0)
    ang = random_angle(n);
    
    one = ang(ang < pi / 4);
    two = ang(ang > pi / 4 & ang < 3 * pi / 4);
    three = ang(ang > 3 * pi / 4 & ang < 5 * pi / 4);
    four = ang(ang > 5 * pi /4);
    
    
    rnd = makedist('Uniform',-d2/2, d2/2);
    first = [repmat(d1/2, 1, length(one)); random(rnd, 1, length(one))];

    rnd = makedist('Uniform',-d1/2, d1/2);
    second = [random(rnd, 1, length(two)) ; repmat(d2/2, 1, length(two))];

    rnd = makedist('Uniform',-d2/2, d2/2);
    third = [repmat(-d1/2, 1, length(one)); random(rnd, 1, length(one))];

    rnd = makedist('Uniform',-d1/2, d1/2);
    fourth = [random(rnd, 1, length(two)) ; repmat(-d2/2, 1, length(two))];

    
    p = [first,second,third,fourth];
end
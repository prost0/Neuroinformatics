function [p] = rectPoints(n, d1, d2, alpha, x0, y0)

    temp = 0:0.025:2*pi;
    ang = temp - pi /4;
    
    one = ang(ang < pi / 4);
    two = ang(ang > pi / 4 & ang < 3 * pi / 4);
    three = ang(ang > 3 * pi / 4 & ang < 5 * pi / 4);
    four = ang(ang > 5 * pi /4);
    
    
    rnd = makedist('Uniform',-d2/2, d2/2);
    first = [repmat(d1/2, 1, length(one)); random(rnd, 1, length(one))];

    rnd = makedist('Uniform',-d1/2, d1/2);
    second = [random(rnd, 1, length(two)) ; repmat(d2/2, 1, length(two))];

    rnd = makedist('Uniform',-d2/2, d2/2);
    third = [repmat(-d1/2, 1, length(three)); random(rnd, 1, length(three))];

    rnd = makedist('Uniform',-d1/2, d1/2);
    fourth = [random(rnd, 1, length(four)) ; repmat(-d2/2, 1, length(four))];

    rot = [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)];
    
    temp = [first,second,third,fourth];
    p = rot * temp + [x0; y0];
end
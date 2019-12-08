n = 100;

d1 = 0.4;
d2 = 0.5;
alpha = -pi/3;


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



points = rectanglos(100,d1,d2,0.5,0.5,0.5);

hold on
plot(points(1,:),points(2,:),'linestyle','none','marker','+','markersize',20)
hold off
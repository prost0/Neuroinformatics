%
%
%   LAB 6
%
%

N = 10;
p = [
    0.5 0.7 0.4 0.6 -0.7 -1.3 0.5 1.3 -0.2 0.7 -1 -0.2;
    0.7 -0.4 -1 -1.5 -1.4 0.9 -0.6 -1.4 -0.4 0.8 -0.1 0.4
    ];
t = [1 1 -1 -1 1 1 -1 1 -1 1 1 1];

tmp = t;
tmp(tmp == -1) = 0;

hold on
plotpv(p,tmp);
hold off

tmp2 = t;
tmp2(tmp2 == 1) = 2;
tmp2(tmp2 == -1) = 1;

tmp2 = ind2vec(tmp2);

net = lvqnet(12,0.1);
net.trainParam.epochs = 300;
net = train(net, p, tmp2);

points = random_points3();
target = sim(net, points);
target = vec2ind(target) - 1;

plotpv(points,target);
point = findobj(gca,'type','line');
set(point,'Color','g');
hold on;
plotpv(p,tmp)
hold off;


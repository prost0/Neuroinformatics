p = [
0.8 0.7 -0.8 -0.9 -0.7 -1.4 -1.1 -1 1.4 -1.4 -0.4 -0.9;
-0.4 1.1 -1.2 -0.5 1.2 0.2 1 0 -0.5 -0.9 -0.5 1.3];
t = [1 1 1 -1 1 -1 1 -1 1 -1 1 1];

tmp1 = t;
tmp1(tmp1 < 0) = 0;
%plotpv(t,t);

t(t > 0) = 2;
t(t <= 0) = 1;


vect = ind2vec(t);

net = lvqnet(12,0.1);
net.trainParam.epochs = 300;
net = train(net, p, vect);

%points = random_points3();
tmp = -1.5:0.1:1.5;
[x,y] = meshgrid(tmp, tmp);
points = [x(:),y(:)]';

target = sim(net, points);
target = vec2ind(target) - 1;

plotpv(points,target);
point = findobj(gca,'type','line');
set(point,'Color','g');
hold on;
plotpv(p,tmp1)
hold off;
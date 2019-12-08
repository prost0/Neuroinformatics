
%
%
%   LAB 6
%
%


bounds = [0 1.5; 0 1.5];   % Cluster centers to be in these bounds.
clusters = 8;          % This many clusters.
points = 10;           % Number of points in each cluster.
std_dev = 0.1;        % Standard deviation of each cluster.
x = nngenc(bounds,clusters,points,std_dev);

net = newsom(bounds, [2, 4]);
net.trainParam.epochs = 150;
net = train(net, x);

y = sim(net,x);
y = vec2ind(y);

P = random_points(5);

t = sim(net, P);
t = vec2ind(t);

hold on

% draw original points
plot(x(1,y == 1), x(2,y == 1),'linestyle','none','marker','o');
plot(x(1,y == 2), x(2,y == 2),'linestyle','none','marker','+');
plot(x(1,y == 3), x(2,y == 3),'linestyle','none','marker','*');
plot(x(1,y == 4), x(2,y == 4),'linestyle','none','marker','.');
plot(x(1,y == 5), x(2,y == 5),'linestyle','none','marker','x');
plot(x(1,y == 6), x(2,y == 6),'linestyle','none','marker','s');
plot(x(1,y == 7), x(2,y == 7),'linestyle','none','marker','d');
plot(x(1,y == 8), x(2,y == 8),'linestyle','none','marker','^');

% draw additional points
plot(P(1,t == 1), P(2,t == 1),'linestyle','none','marker','o','markersize',20);
plot(P(1,t == 2), P(2,t == 2),'linestyle','none','marker','+','markersize',20);
plot(P(1,t == 3), P(2,t == 3),'linestyle','none','marker','*','markersize',20);
plot(P(1,t == 4), P(2,t == 4),'linestyle','none','marker','.','markersize',20);
plot(P(1,t == 5), P(2,t == 5),'linestyle','none','marker','x','markersize',20);
plot(P(1,t == 6), P(2,t == 6),'linestyle','none','marker','s','markersize',20);
plot(P(1,t == 7), P(2,t == 7),'linestyle','none','marker','d','markersize',20);
plot(P(1,t == 8), P(2,t == 8),'linestyle','none','marker','^','markersize',20);

plotsom(net.IW{1, 1}, net.layers{1}.distances)


hold off

disp(y);

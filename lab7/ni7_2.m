
h = 0.025;
phi = 0.01:h:2*pi;


x = arrayfun(@(a) (1 / sqrt(a)) * cos(a) , phi);
y = arrayfun(@(a) (1 / sqrt(a)) * sin(a) , phi);
p = [x; y];



net = feedforwardnet([10, 1, 10]);
net = init(net);
net = configure(net, p, p);

net.trainParam.epochs = 10000;
net.trainParam.goal = 1e-5;
net.trainParam.max_fail = 1000;

net = train(net, p, p);

y = sim(net, p);

hold on
plot(p(1,:),p(2,:), 'marker','.', 'markersize',20, 'color','b')
plot(y(1,:),y(2,:), 'marker','.','markersize',10,'color','r')
hold off
h = 0.025;
phi = 0.01:h:11*pi/12;

x = arrayfun(@(a) (1 / sqrt(a)) * cos(a) , phi);
y = arrayfun(@(a) (1 / sqrt(a)) * sin(a) , phi);
p = [x; y; phi];


net = feedforwardnet([10, 2, 10]);
net = init(net);
net = configure(net, p, p);

net.trainParam.epochs = 10000;
net.trainParam.goal = 1e-5;
net.trainParam.max_fail = 1000;

net = train(net, p, p);

y = sim(net, p);


figure
plot3(p(1,:),p(2,:),p(3,:),'markersize',20,'color','b')
figure
plot3(y(1,:),y(2,:),y(3,:),'markersize',20,'color','r')

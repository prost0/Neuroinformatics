n = 100;

d1 = 0.3;
d2 = 0.5;
x0 = 0.4;
y0 = -0.2;
alpha = pi/3;


points = rectPoints(100,d1,d2,alpha,x0,y0);
 
 %hold on
 %plot(points(1,:),points(2,:),'linestyle','none','marker','.','markersize',20)
% hold off

% p = con2seq(points);

p = points;
net = feedforwardnet(1);
% net = init(net);
net = configure(net, p, p);

net.layers{1}.transferFcn = 'purelin';
net.layers{2}.transferFcn = 'purelin';
net.trainParam.epochs = 100;
net.trainParam.goal = 1e-5;
net.trainParam.max_fail = 100;

net = train(net, p, p);

y = sim(net, p);
% y = seq2con(y);
% y = y{1};

hold on
plot(y(1,:),y(2,:),'marker','.','markersize',15, 'color', 'b')
plot(points(1,:),points(2,:),'linestyle', 'none','marker','.','markersize',15,'color','r')
hold off

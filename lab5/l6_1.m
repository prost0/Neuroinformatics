clc
 
X=[0 1.5; 0 1.5];
clusters=8;
points=10;    %3адание количества точек в кластере
std_dev=0.1;
 
p=nngenc(X,clusters,points,std_dev);
net = competlayer(8);
net.trainParam.epochs = 50;

net = train(net,p);
y = sim(net, p);
y = vec2ind(y);

rnd = makedist('Uniform',0,1.5);
P = random(rnd,2,5);

t = sim(net,P);
t = vec2ind(t);

IW = net.IW{1};

hold on
    % draw original points
    plot(p(1,y == 1), p(2,y == 1),'linestyle','none','marker','o');
    plot(p(1,y == 2), p(2,y == 2),'linestyle','none','marker','+');
    plot(p(1,y == 3), p(2,y == 3),'linestyle','none','marker','*');
    plot(p(1,y == 4), p(2,y == 4),'linestyle','none','marker','.');
    plot(p(1,y == 5), p(2,y == 5),'linestyle','none','marker','p');
    plot(p(1,y == 6), p(2,y == 6),'linestyle','none','marker','<');
    plot(p(1,y == 7), p(2,y == 7),'linestyle','none','marker','d');
    plot(p(1,y == 8), p(2,y == 8),'linestyle','none','marker','^');
    
    % draw additional points
    plot(P(1,t == 1), P(2,t == 1),'linestyle','none','marker','o','markersize',20);
    plot(P(1,t == 2), P(2,t == 2),'linestyle','none','marker','+','markersize',20);
    plot(P(1,t == 3), P(2,t == 3),'linestyle','none','marker','*','markersize',20);
    plot(P(1,t == 4), P(2,t == 4),'linestyle','none','marker','.','markersize',20);
    plot(P(1,t == 5), P(2,t == 5),'linestyle','none','marker','x','markersize',20);
    plot(P(1,t == 6), P(2,t == 6),'linestyle','none','marker','<','markersize',20);
    plot(P(1,t == 7), P(2,t == 7),'linestyle','none','marker','d','markersize',20);
    plot(P(1,t == 8), P(2,t == 8),'linestyle','none','marker','^','markersize',20);
    
    % draw centers
    plot(IW(:,1),IW(:,2),'marker','s','markersize',30, 'linestyle','none','color','black');
    
    %plotsom(net.IW{1, 1}, net.layers{1}.distances)
    
hold off

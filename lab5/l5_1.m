
p1_args = 0:0.025:1;
p1_values = sin(4 * pi * p1_args);
t1 = ones(1,length(p1_args));
p2_args =  0.78:0.025:2.35;
p2_values = 1.5 * sin(-5 * power(p2_args,2) + 10 * p2_args - 5) + 0.4;
t2 = -1 * ones(1, length(p2_args));


R = [2 2 5];

P = [repmat(p1_values,1,5),p2_values,repmat(p1_values,1,3),p2_values,repmat(p1_values,1,3),p2_values];
T = [repmat(t1,1,5),t2,repmat(t1,1,3),t2,repmat(t1,1,3),t2];

P = con2seq(P);
T = con2seq(T);

net = layrecnet(1:2,8,'trainoss');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net = configure(net, P, T);

[Ps,Pi,Ai,Ts] = preparets(net,P,T);

net.trainParam.epochs = 100;
net.trainParam.goal = 1e-5;

% view(net)

net = train(net, Ps, Ts, Pi, Ai);

Y = sim(net, Ps);
Y = cell2mat(Y);

Y(Y >= 0) = 1;
Y(Y < 0) = -1;
 
Ps = cell2mat(Ps);
Ts = cell2mat(Ts);

hold on
    plot(1:1:(length(Ts)), Ts, 'color','r')
    plot(1:1:(length(Ts)), Y, 'color','g')
hold off

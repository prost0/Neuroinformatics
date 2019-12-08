t0 = 0;
tn = 10;
dt = 0.01;
n = (tn - t0) / dt + 1;
fun = @(k)cos(k.^2-2.*k + 3);
fun2 = @(y, u)y ./ (1 + y.^2) + u.^3;

u = zeros(1, n);
y = zeros(1, n);
u(1) = fun(0);

for i = 2 : n
    t = t0 + (i - 1) * dt;
    y(i) = fun2(y(i - 1), u(i - 1));
    u(i) = fun(t);
end

figure
subplot(2, 1, 1)
plot(t0 : dt : tn, u, '-b'), grid
ylabel('control')

subplot(2, 1, 2)
plot(t0 : dt : tn, y, '-r'), grid
ylabel('state')
xlabel('t')
x = u;
D = 3;

ntrain = 700;
nval = 200;
ntest = 97;

trainInd = 1 : ntrain;
valInd = ntrain + 1 : ntrain + nval;
testInd = ntrain + nval + 1 : ntrain + nval + ntest;

net = narxnet(1 : 3, 1 : 3, 10);
net.trainFcn = 'trainlm';
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';

net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;
net = init(net);
net.trainParam.epochs = 600;
net.trainParam.max_fail = 600;
net.trainParam.goal = 1.0e-8;

[Xs, Xi, Ai, Ts] = preparets(net, con2seq(x), {}, con2seq(y));
net = train(net, Xs, Ts, Xi, Ai);
Y = sim(net, Xs, Xi, Ai); 

figure
subplot(3, 1, 1)
plot(t0 : dt : tn, u, '-b'),grid 
ylabel('Answer') 

subplot(3, 1, 2) 
plot(t0 : dt : tn, x, '-b', t0 : dt : tn, [x(1:D) cell2mat(Y)], '-r'), grid 
ylabel('Out') 

subplot(3, 1, 3) 
plot(t0+D*dt : dt : tn, x(D+1:end) - cell2mat(Y)), grid 
ylabel('Error(train)') 

last_elem_x = x(valInd(length(valInd)-2 : length(valInd)));
last_elem_u = u(valInd(length(valInd)-2 : length(valInd)));
inp = [con2seq(u(testInd)); con2seq(x(testInd))];
delay = [con2seq(last_elem_u); con2seq(last_elem_x)];

out_for_test = sim(net, inp, delay, Ai);
figure;
hold on;
plot(x(testInd), '.-b')
plot(cell2mat(out_for_test), '-or');
grid minor;
grid on;

figure;
hold on;
errortst = cell2mat(out_for_test) - x(testInd);
plot(errortst, '-r');
legend('Error (test)');
grid minor;
grid on;
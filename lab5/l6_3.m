N = 20;
T = -1.5 + (1.5 + 1.5) * rand(2, N);
hold on
%plot(T(1,:),T(2,:), '-V','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',7), grid;
hold off

net = newsom(T, N);
net.trainParam.epochs = 600;
net = train(net, T);
hold on
plotsom(net.IW{1, 1}, net.layers{1}.distances),
plot(T(1,:),T(2,:), '-V','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',7), grid;
hold off
%
%
%   LAB 6
%
%
N = 20;
p = random_points2(N);

% hold on
% plot(p(1,:),p(2,:), '-V','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',7), grid
% hold off

net = newsom(p, N);
net.trainParam.epochs = 600;
net = train(net, p);

hold on
plotsom(net.IW{1, 1}, net.layers{1}.distances)
plot(p(1,:),p(2,:), '-V','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',7), grid
hold off


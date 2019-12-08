date = table{:, 1};
answ = table{:, 2};

D = 0:5;
hiddenLayerNeurons = 8;
answ = smooth(answ, 12);

x = date(1:500);

net = timedelaynet(D, hiddenLayerNeurons);

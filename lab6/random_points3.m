function P = random_points3()
    tmp = -1.5:0.1:1.5;
    [x,y] = meshgrid(tmp, tmp);
    P = [x(:),y(:)]';
end
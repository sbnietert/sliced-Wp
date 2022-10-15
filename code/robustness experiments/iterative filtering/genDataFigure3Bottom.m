clear;
eps = 0.1;
tau = 0.1;
cher = 2.5;

ds = [10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200];

for d = ds
    N = 10*floor(d/eps^2);
    fprintf('Training with dimension = %d, number of samples = %d \n', d, round(N, 0))

    X = zeros(round((1-eps)*N),d);
    Y = mvnrnd(zeros(1,d), eye(d), round(eps*N));
    Ynorm = vecnorm(Y');
    Y = bsxfun(@rdivide,Y,Ynorm(:));
    X = [X; Y*sqrt(d)/eps];

    fprintf('Saving clean (and corrupted - null contamination) samples...');
    save(strcat('data2/clean',num2str(d)),'X')
    
    fprintf('Filtering')
    filteredData = outlierFilter(X, eps, tau, cher);
    fprintf('Saving filtered samples...');
    save(strcat('data2/filtered',num2str(d)),'filteredData')
    fprintf('...done\n')
end
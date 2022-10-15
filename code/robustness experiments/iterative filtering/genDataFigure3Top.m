clear;
eps = 0.1;
tau = 0.1;
cher = 2.5;

ds = [10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200];

for d = ds
    N = 10*floor(d/eps^2);
    fprintf('Training with dimension = %d, number of samples = %d \n', d, round(N, 0))

    X =  mvnrnd(zeros(1,d), eye(d), round((1-eps)*N)) + ones(round((1-eps)*N), d);

    fprintf('Saving clean samples');
    save(strcat('data/clean',num2str(d)),'X')
    
    Y1 = randi([0 1], round(0.5*eps*N), d); 
    Y2 = [12*ones(round(0.5*eps*N),1), -2 * ones(round(0.5*eps*N), 1), zeros(round(0.5 * eps * N), d-2)];
    X = [X; Y1; Y2];

    fprintf('Saving corrupted samples');
    save(strcat('data/corrupted',num2str(d)),'X')
    
    fprintf('Filtering');
    filteredData = outlierFilter(X, eps, tau, cher);

    fprintf('Saving filtered samples')
    save(strcat('data/filtered',num2str(d)),'filteredData')
end

% This script prints the result saved in result.mat

%% Load result data from result.mat
load result.mat

%% Plot the history of log-likelihood
figure;
plot(Lt);
xlabel('Number of iterations');
ylabel('Log-likelihood');

%% Find keywords for each topic
n_kw = 10; % Find 10 keywords
for z = 1:n_z
    fprintf('Key words for topic %d:\n', z);
    [S, I] = sort(p_w_z(:,z), 'descend');
    for w = I(1:n_kw)'
        fprintf('%d %s\t(%f)\n', w, words{w}, p_w_z(w,z))
    end
    fprintf('\n')
end
fprintf('\n')

%% Randomly pick documents and show their p(z|d) for each z
n_d_show = 10; % Pick 10 documents
for d = sort(randsample(size(n_dw,1), n_d_show))'
    fprintf('Topic weights for document %d:\n', d);
    for z = 1:n_z
        fprintf('%f\t', p_z_d(z,d))
    end
    fprintf('\n')
end
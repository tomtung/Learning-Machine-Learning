% This script prints the result saved in result.mat
load result.mat
n_kw = 10; % find 10 keywords for each topic
for z = 1:n_z
    fprintf('Key words for topic %d:\n', z);
    [S, I] = sort(p_w_given_z(:,z), 'descend');
    for w = I(1:n_kw)'
        fprintf('%d %s\t(%f)\n', w, words{w}, p_w_given_z(w,z))
    end
    fprintf('\n')
end
fprintf('\n')

n_d_show = 10; % show p(z|d) for first 10 documents
for d = sort(randsample(size(n_dw,1), n_d_show))'
    fprintf('Topic weights for document %d:\n', d);
    for z = 1:n_z
        fprintf('%f\t', p_z_given_d(z,d))
    end
    fprintf('\n')
end
% Given the number of occurrence of word w in document d n_dw(d,w), and the number of topics to discover n_z, this function returns p(w|z) and p(z|d)
function [p_w_given_z, p_z_given_d] = pLSA(n_dw, n_z)

% filter out words that are too common or too obscure
for w = 1:size(n_dw,2)
    if size(nonzeros(n_dw(:,w)),1) > size(n_dw,1)*1.5/n_z || size(nonzeros(n_dw(:,w)),1) <= size(n_dw,1)/(n_z*10)
        n_dw(:,w) = 0;
    end
end

% pre-allocate space
[n_d, n_w] = size(n_dw); % max indices of d and w
p_z_given_d = rand(n_z, n_d); % p(z|d)
p_w_given_z = rand(n_w, n_z); % p(w|z)
n_p_z_given_dw = cell(n_z, 1); % n(d,w) * p(z|d,w)
for z = 1:n_z
    n_p_z_given_dw{z} = sprand(n_dw);
end

p_dw = sprand(n_dw); % p(d,w)
L = intmin; % log-likelihood
improvement = intmax; % used to determine convergence
while improvement > 0.001*abs(L)
    disp('E-step');
    for d = 1:n_d
        for w = find(n_dw(d,:))
            for z = 1:n_z
                n_p_z_given_dw{z}(d,w) = p_z_given_d(z,d) * p_w_given_z(w,z) * n_dw(d,w) / p_dw(d, w);
            end
        end
    end
    
    disp('M-step');
    disp('update p(z|d)')
    concat = cat(2, n_p_z_given_dw{:}); % make n_p_z_given_dw{:}(d,:)) possible
    for d = 1:n_d
        for z = 1:n_z
            p_z_given_d(z,d) = sum(n_p_z_given_dw{z}(d,:));
        end
        p_z_given_d(:,d) = p_z_given_d(:,d) / sum(concat(d,:));
    end
    
    disp('update p(w|z)')
    for z = 1:n_z
        for w = 1:n_w
            p_w_given_z(w,z) = sum(n_p_z_given_dw{z}(:,w));
        end
        p_w_given_z(:,z) = p_w_given_z(:,z) / sum(n_p_z_given_dw{z}(:));
    end

    % update p(d,w) and calculate likelihood to determine convergence
    prevL = L; L = 0;
    for d = 1:n_d
        for w = find(n_dw(d,:))
            p_dw(d,w) = 0;
            for z = 1:n_z
                p_dw(d,w) = p_dw(d,w) + p_w_given_z(w,z) * p_z_given_d(z,d);
            end
            L = L + n_dw(d,w) * log(p_dw(d, w));
        end
    end
    improvement = L - prevL;

    fprintf('L = %f, improvement = %f%%\n', L, improvement/abs(L)*100);
end
disp('Improved less than 0.1%. Algorithm converged.')
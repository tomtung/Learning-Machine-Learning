% This script prepares the data, trains the model, saves and prints the result

if ~exist('prepared_data.mat', 'file')
    data = 'data.txt';
    disp('Preparing data ...')
    prepare(data);
end
load prepared_data.mat

n_z = 15; % number of topics to discover
disp('Training pLSA model ...')
[p_w_given_z, p_z_given_d] = pLSA(n_dw, n_z);
save result.mat word2Index words n_dw n_z p_w_given_z p_z_given_d

show_result
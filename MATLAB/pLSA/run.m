% This script prepares the data, trains the model, saves and prints the result

%% Clear and Close Figures
clear all; close all; clc

%% Prepare data
if ~exist('prepared_data.mat', 'file')
    prepare;
end
load prepared_data.mat

%% Train the pLSA model and save the result
n_z = 10; % number of topics to discover
disp('Training pLSA model ...')
[p_w_z, p_z_d, Lt] = pLSA(n_dw, n_z, 200);
save result.mat word2Index words n_dw n_z p_w_z p_z_d Lt

%% Show the result
show_result
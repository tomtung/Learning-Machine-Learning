% This function reads data, prepare and save it to prepared_data.mat
function [] = prepare(data)
word2Index = containers.Map;
words = cell(50000, 1);
n_dw = spalloc(1000000, 500000, 50000000);
nDoc = 0;

% Read data and get prepared
fid = fopen(data);
while ~feof(fid)
    nDoc = nDoc + 1;
    fprintf('Loading document %d\n', nDoc)
    line = regexprep(lower(fgetl(fid)), '[^\w ]+', ' ');
    wordsInLine = textscan(line, '%s');
    for i = 1:size(wordsInLine{1}, 1), word = wordsInLine{1}{i};
        if length(word) < 4, continue; end
        if ~isKey(word2Index, word)
            w = size(word2Index, 1) + 1;
            word2Index(word) = w;
            words{w} = word;
        else
            w = word2Index(word);
        end
        n_dw(nDoc, w) = n_dw(nDoc, w) + 1;
    end
end
fclose(fid);

n_dw = n_dw(1:nDoc,1:size(word2Index));
words = words(1:size(word2Index));

save prepared_data.mat word2Index words n_dw
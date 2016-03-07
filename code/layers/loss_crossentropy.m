% ======================================================================
% Matrix size reference:
% ----------------------------------------------------------------------
% input: num_nodes * batch_size
% labels: batch_size * 1
% ======================================================================

function [loss, dv_input] = loss_crossentropy(input, labels, hyper_params, backprop)

assert(max(labels) <= size(input,1));
[num_nodes ,batch_size] = size(input);

% calculate loss
loss = 0;
mask = zeros(size(input));
for i=1:batch_size
    mask(labels(i),i) = 1;
end
mask = logical(mask);
loss = sum(-log(input(mask))); 

dv_input = zeros(size(input));
if backprop
    % calculate dL/dx
    dv_input(mask) = -1 ./ input(mask); 
end
end

function loss = crossentropy(x, label)
    % label in range 0-9, verbose function
    loss = -log(x(label));
end
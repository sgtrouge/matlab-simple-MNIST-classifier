% input: batch_size * num_in_nodes 
% output: output of last layer
% activations: outputs of all layers

function [output,activations] = inference(model,input)
% Do forward propagation through the network to get the activation
% at each layer, and the final output

num_layers = numel(model.layers);
activations = cell(num_layers,1);

if num_layers == 0
    return;
end
% FORWARD PROPAGATION CODE
layer = model.layers(1);
[activations{1},~,~] = layer.fwd_fn(input, layer.params, layer.hyper_params, false, []);
for i=2:num_layers
    layer = model.layers(i);
    [activations{i},~,~] = layer.fwd_fn(activations{i-1}, layer.params, layer.hyper_params, false, []);
end

output = activations{end};

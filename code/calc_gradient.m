function [grad] = calc_gradient(model, input, activations, dv_output)
% Calculate the gradient at each layer, to do this you need dv_output
% determined by your loss function and the activations of each layer.
% The loop of this function will look very similar to the code from
% inference, just going in reverse this time.

num_layers = numel(model.layers);
grad = cell(num_layers,1);

if num_layers == 0
    return;
end
% Determine the gradient at each layer with weights to be updated
for i=num_layers:-1:2
    layer = model.layers(i);
    [~,dv_output,grad{i}] = layer.fwd_fn(activations{i-1}, layer.params, layer.hyper_params, true, dv_output);
end
layer = model.layers(1);
[~,~,grad{1}] = layer.fwd_fn(input, layer.params, layer.hyper_params, true, dv_output);

end
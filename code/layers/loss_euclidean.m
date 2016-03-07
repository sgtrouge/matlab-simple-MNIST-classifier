% ======================================================================
% Matrix size reference:
% ----------------------------------------------------------------------
% input: any dimension * batch_size
% labels: same size as input
% ======================================================================

function [loss, dv_input] = loss_euclidean(input, labels, hyper_params, backprop)

assert(isequal(size(labels),size(input)));

diff = input - labels;
loss = sum(diff(:)'*diff(:))/(2*length(diff(:)));

dv_input = [];
if backprop
	dv_input = diff/length(diff(:));
end


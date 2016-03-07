function [output, dv_input, grad] = fn_sigmoid(input, params, hyper_params, backprop, dv_output)

output = 1 ./ (exp(-input) + 1);

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
		dv_input = output.*(1-output).*dv_output;
end
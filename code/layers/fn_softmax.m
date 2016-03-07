% ======================================================================
% Matrix size reference:
% ----------------------------------------------------------------------
% input: num_classes * batch_size
% output: num_classes * batch_size
% ======================================================================

function [output, dv_input, grad] = fn_softmax(input, params, hyper_params, backprop, dv_output)

[num_classes,batch_size] = size(input);
output = zeros(num_classes, batch_size);
% forward code, max(input) was used to avoid overflow
% output = (exp(input-repmat(max(input),[size(input,1),1]))) ./ repmat(sum(exp(input-repmat(max(input),[size(input,1),1])),1), [size(input, 1), 1]);
output = exp(input) ./ repmat(sum(exp(input),1), [size(input,1), 1]);

dv_input = [];

% This is included to maintain consistency in the return values of layers,
% but there is no gradient to calculate in the softmax layer since there
% are no weights to update.
grad = struct('W',[],'b',[]); 


if backprop
    dv_input = zeros(size(input));
    % BACKPROP CODE
    M = zeros(num_classes);
    for b=1:batch_size
        for i=1:num_classes
            for j=1:num_classes
                if i==j
                    M(i,j) = output(i,b) - output(i,b) .* output(j,b);
                else 
                    M(i,j) = - output(i,b) .* output(j,b);
                end
            end
        end
        dv_input(:,b) = M'*dv_output(:,b); 
    end
end

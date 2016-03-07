% ======================================================================
% Matrix size reference:
% input: in_height * in_width * num_channels * batch_size
% output: out_height * out_width * num_filters * batch_size
% hyper parameters: (used for options like stride, padding (neither is required for this project))
% params.W: filter_height * filter_width * filter_depth * num_filters
% params.b: num_filters * 1
% dv_output: same as output
% dv_input: same as input
% grad.W: same as params.W
% grad.b: same as params.b
% ======================================================================

function [output, dv_input, grad] = fn_conv(input, params, hyper_params, backprop, dv_output)

[in_height,in_width,num_channels,batch_size] = size(input);
[filter_height,filter_width,filter_depth,num_filters] = size(params.W);

out_height = size(input,1) - size(params.W,1) + 1; 
out_width = size(input,2) - size(params.W,2) + 1; 
output = zeros(out_height,out_width,num_filters,batch_size);
% TODO: FORWARD CODE

for i = 1:batch_size
    for j = 1:num_filters
        W = params.W(:,:,:,j);
        b = params.b(j,1);
        output(:,:,j,i) = b*ones(out_height,out_width); 
        for k = 1:num_channels
            output(:,:,j,i) = output(:,:,j,i)+conv2(input(:,:,k,i),W(:,:,k),'valid');
        end
    end
end

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
    dv_input = zeros(size(input));
    grad.W = zeros(size(params.W));
    grad.b = zeros(size(params.b));
    for i = 1:batch_size 
        for k = 1:num_channels
            for j = 1:num_filters
                dv_input(:,:,k,i) = dv_input(:,:,k,i)+conv2(dv_output(:,:,j,i),rot90(params.W(:,:,k,j),2),'full');
            end
        end
    end
    
    for j = 1:num_filters
        for k = 1:num_channels
            grad.W(:,:,k,j) = zeros(filter_height, filter_width);
            for i = 1:batch_size
                grad.W(:,:,k,j) = grad.W(:,:,k,j)+conv2(rot90(input(:,:,k,i),2),dv_output(:,:,j,i),'valid');
            end
        end
        
        grad.b(j,1)=0;
        for i=1:batch_size
            grad.b(j,1) = grad.b(j,1)+sum(sum(dv_output(:,:,j,i)));
        end
    end
end

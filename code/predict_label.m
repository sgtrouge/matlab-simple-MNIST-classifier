function [prediction, accuracy, loss]=predict_label(model, input, label)
    [output, activations] = inference(model, input);
    [~, prediction] = max(output);
    prediction = prediction';
    accuracy = sum(prediction==label)/length(label);
    
    % calculate test loss
    [loss,~] = loss_crossentropy(output, label,[], false);
end
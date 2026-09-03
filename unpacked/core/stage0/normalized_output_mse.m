function values = normalized_output_mse(Y, Yhat)
%NORMALIZED_OUTPUT_MSE Variance-normalized MSE for each output dimension.

    if isempty(Y)
        values = [];
        return;
    end
    if ~isequal(size(Y), size(Yhat)) || any(~isfinite(Yhat(:)))
        values = Inf(1,size(Y,2));
        return;
    end
    values = zeros(1,size(Y,2));
    for k = 1:size(Y,2)
        denom = mean((Y(:,k) - mean(Y(:,k))).^2);
        denom = max(denom, 1e-12 * max(1, mean(Y(:,k).^2)));
        values(k) = mean((Y(:,k) - Yhat(:,k)).^2) / denom;
    end
end

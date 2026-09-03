function T = soft_saturated_lorenz96_noise_rows_to_table(rows)
%SOFT_SATURATED_LORENZ96_NOISE_ROWS_TO_TABLE Flatten robustness rows for CSV.
%
% The dedicated Lorenz--96 noise ablation reports clean held-out ID-test
% vector-field RMSE only. OOD and rollout metrics are deliberately omitted.

    if isempty(rows); T = table(); return; end
    n = numel(rows);
    noiseLevel = reshape([rows.noiseLevel],[],1);
    noisePercent = 100*noiseLevel;
    roundIndex = reshape([rows.roundIndex],[],1);
    nTrain = reshape([rows.nTrain],[],1);
    method = string({rows.method}).';
    derivativeRMSE = reshape([rows.derivativeRMSE],[],1);
    validationMSE = reshape([rows.validationMSE],[],1);
    trainTime = reshape([rows.trainTime],[],1);
    activeCoefficients = reshape([rows.activeCoefficients],[],1);
    noiseSeed = nan(n,1);
    for k = 1:n
        if isfield(rows(k),'noiseSeed') && ~isempty(rows(k).noiseSeed)
            noiseSeed(k) = rows(k).noiseSeed;
        end
    end
    T = table(noiseLevel,noisePercent,roundIndex,nTrain,method,noiseSeed, ...
        derivativeRMSE,validationMSE,trainTime,activeCoefficients);
end

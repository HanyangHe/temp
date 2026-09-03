function T = single_generator_dynamic_noise_rows_to_table(rows)
%SINGLE_GENERATOR_DYNAMIC_NOISE_ROWS_TO_TABLE Flatten robustness rows for CSV.
%
% The dedicated noise ablation reports clean ID-test vector-field NRMSE only
% as its accuracy metric.  OOD and rollout quantities are deliberately not
% exported here because they are outside the scope of this robustness test.

    if isempty(rows); T = table(); return; end
    n = numel(rows);
    noiseLevel = reshape([rows.noiseLevel],[],1);
    noisePercent = 100*noiseLevel;
    roundIndex = reshape([rows.roundIndex],[],1);
    nTrain = reshape([rows.nTrain],[],1);
    method = string({rows.method}).';
    derivativeNRMSE = reshape([rows.derivativeNRMSE],[],1);
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
        derivativeNRMSE,validationMSE,trainTime,activeCoefficients);
end

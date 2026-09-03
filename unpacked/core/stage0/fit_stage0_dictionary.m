function model = fit_stage0_dictionary(dictionary, data, options)
%FIT_STAGE0_DICTIONARY Select STLSQ threshold by multi-output validation loss.

    PhiTr = dictionary.PhiTr;
    PhiVal = dictionary.PhiVal;
    computeHoldout = logical(get_option_local(options, 'computeHoldoutMetrics', true));
    storePredictions = logical(get_option_local(options, 'storePredictions', true));
    buildExpressions = logical(get_option_local(options, 'buildExpressions', true));
    if computeHoldout
        PhiTe = dictionary.PhiTe;
        PhiOod = dictionary.PhiOod;
    else
        PhiTe = zeros(0, size(PhiTr,2));
        PhiOod = [];
    end

    scale = sqrt(mean(PhiTr.^2, 1));
    constantRows = false(1, size(PhiTr,2));
    for j = 1:numel(dictionary.termNames)
        constantRows(j) = strcmp(strrep(dictionary.termNames{j}, ' ', ''), '1');
    end
    scale(constantRows) = 1;
    scale(~isfinite(scale) | scale < options.scaleFloor) = 1;
    TrN = PhiTr ./ scale;
    ValN = PhiVal ./ scale;
    if computeHoldout
        TeN = PhiTe ./ scale;
        if isempty(PhiOod); OodN = []; else; OodN = PhiOod ./ scale; end
    else
        TeN = [];
        OodN = [];
    end

    thresholds = unique(options.thresholdList(:).');
    thresholds = thresholds(isfinite(thresholds) & thresholds >= 0);
    if isempty(thresholds)
        thresholds = 0;
    end

    bestScore = Inf;
    best = struct();
    for i = 1:numel(thresholds)
        [XiN, activeMask] = stlsq_multioutput(TrN, data.Ytr, thresholds(i), options);
        YtrHat = TrN * XiN;
        YvalHat = ValN * XiN;
        valPerOutput = normalized_output_mse(data.Yval, YvalHat);
        valLossMean = mean(valPerOutput);
        valLossWorst = max(valPerOutput);
        worstWeight = get_option_local(options, 'worstOutputWeight', 0);
        valLoss = valLossMean + worstWeight * valLossWorst;
        nActive = nnz(activeMask);
        score = valLoss + options.complexityTieWeight * nActive;
        if isfinite(score) && score < bestScore
            bestScore = score;
            best.XiN = XiN;
            best.activeMask = activeMask;
            best.threshold = thresholds(i);
            best.valLoss = valLoss;
            best.valLossMean = valLossMean;
            best.valLossWorst = valLossWorst;
            best.valPerOutput = valPerOutput;
            best.YtrHat = YtrHat;
            best.YvalHat = YvalHat;
        end
    end

    if isempty(fieldnames(best))
        error('No finite STLSQ solution was found for the current Stage-0 dictionary.');
    end

    Xi = best.XiN ./ scale.';
    YtrHat = PhiTr * Xi;
    YvalHat = PhiVal * Xi;
    if computeHoldout
        YteHat = TeN * best.XiN;
        if isempty(OodN); YoodHat = []; else; YoodHat = OodN * best.XiN; end
    else
        YteHat = [];
        YoodHat = [];
    end

    model = struct();
    model.termNames = dictionary.termNames(:);
    model.Xi = Xi;
    model.XiScaled = best.XiN;
    model.scale = scale;
    model.threshold = best.threshold;
    model.activeMask = logical(best.activeMask);
    model.nActiveCoefficients = nnz(model.activeMask);
    model.nActiveTerms = nnz(any(model.activeMask,2));
    model.trainMetrics = compute_regression_metrics(YtrHat, data.Ytr);
    model.valMetrics = compute_regression_metrics(YvalHat, data.Yval);
    model.trainPerOutputNormalizedMSE = normalized_output_mse(data.Ytr, YtrHat);
    model.valPerOutputNormalizedMSE = normalized_output_mse(data.Yval, YvalHat);
    model.trainPerOutputMSE = mean((YtrHat - data.Ytr).^2, 1);
    model.valPerOutputMSE = mean((YvalHat - data.Yval).^2, 1);
    model.valLossMean = mean(model.valPerOutputNormalizedMSE);
    model.valLossWorst = max(model.valPerOutputNormalizedMSE);
    model.valLoss = model.valLossMean + get_option_local(options, 'worstOutputWeight', 0) * model.valLossWorst;

    if computeHoldout
        model.testMetrics = compute_regression_metrics(YteHat, data.Yte);
        model.testPerOutputNormalizedMSE = normalized_output_mse(data.Yte, YteHat);
        model.testPerOutputMSE = mean((YteHat - data.Yte).^2, 1);
        if ~isempty(YoodHat) && isfield(data,'Yood') && ~isempty(data.Yood)
            model.oodMetrics = compute_regression_metrics(YoodHat, data.Yood);
            model.oodPerOutputNormalizedMSE = normalized_output_mse(data.Yood, YoodHat);
            model.oodPerOutputMSE = mean((YoodHat - data.Yood).^2, 1);
        else
            model.oodMetrics = empty_metrics_local();
            model.oodPerOutputNormalizedMSE = [];
            model.oodPerOutputMSE = [];
        end
    else
        model.testMetrics = empty_metrics_local();
        model.testPerOutputNormalizedMSE = [];
        model.testPerOutputMSE = [];
        model.oodMetrics = empty_metrics_local();
        model.oodPerOutputNormalizedMSE = [];
        model.oodPerOutputMSE = [];
    end

    if storePredictions
        model.prediction = struct('Ytr',YtrHat,'Yval',YvalHat,'Yte',YteHat,'Yood',YoodHat);
    else
        model.prediction = struct('Ytr',[],'Yval',[],'Yte',[],'Yood',[]);
    end
    if buildExpressions
        model.outputExpressions = stage0_model_to_expressions( ...
            model.termNames, Xi, options.coefficientZeroTolerance, model.activeMask);
    else
        model.outputExpressions = {};
    end
    model.complexity = model.nActiveCoefficients;
    model.fitness = model.valLoss;
    model.holdoutMetricsComputed = computeHoldout;
end

function metrics = empty_metrics_local()
    metrics = struct('mse',NaN,'rmse',NaN,'mae',NaN,'r2',NaN);
end

function value = get_option_local(options, name, defaultValue)
    if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
        value = options.(name);
    else
        value = defaultValue;
    end
end

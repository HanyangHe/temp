function [predictFcn, info] = make_system_identification_predictor(methodName, methodResult)
%MAKE_SYSTEM_IDENTIFICATION_PREDICTOR Return a row-sample vector-field predictor.

    methodName = lower(strtrim(char(methodName)));
    predictFcn = [];
    info = struct('available',false,'reason','','method',methodName);

    switch methodName
        case 'phdn'
            if ~isfield(methodResult,'Coef_M_est') || isempty(methodResult.Coef_M_est) || ...
                    ~isfield(methodResult,'arch') || isempty(methodResult.arch)
                info.reason = ['PhDN coefficient/architecture fields are unavailable. ', ...
                    'The SMIB demo disables the Stage-0 direct-bypass route so this ', ...
                    'normally indicates an incomplete result.'];
                return;
            end
            d = methodResult.data;
            useIONorm = false;
            useLayerNorm = false;
            if isfield(methodResult,'opts') && isfield(methodResult.opts,'norm')
                useIONorm = logical(getfield_default_local(methodResult.opts.norm, ...
                    'useInputOutputNorm', false));
                useLayerNorm = logical(getfield_default_local(methodResult.opts.norm, ...
                    'useLayerNorm', false));
            end
            normOpt = fit_norm_options(d.Xtr, d.Ytr, useIONorm, useLayerNorm);
            predictFcn = @(X) model_forward(X, methodResult.Coef_M_est, ...
                methodResult.arch, normOpt);
            info.available = true;

        case {'stage0-sr','stage0sr'}
            expressions = getfield_default_local(methodResult, ...
                'selectedExpressions', {});
            if isempty(expressions)
                info.reason = ['Stage0-SR selected expressions are unavailable; ', ...
                    'derivative/test metrics remain available.'];
                return;
            end
            predictFcn = @(X) predict_single_generator_dynamic_stage0_sr( ...
                methodResult, X);
            info.available = true;
            info.reason = ['Arbitrary-state predictor evaluates the selected ', ...
                'Stage-0 SINDy/PySR expressions directly.'];

        case 'sindy'
            predictFcn = @(X) predict_sindy_baseline(methodResult, X);
            info.available = true;

        case 'mlp'
            if isfield(methodResult,'net') && ~isempty(methodResult.net)
                predictFcn = @(X) methodResult.net(X.').';
                info.available = true;
            else
                info.reason = ['Arbitrary-state rollout requires the fixed_fitnet MLP ', ...
                    'protocol. The external Python sweep currently exports only split ', ...
                    'predictions, not a MATLAB-callable model.'];
            end

        otherwise
            info.reason = sprintf(['No arbitrary-state predictor is registered for %s. ', ...
                'Derivative/test metrics remain available.'], methodName);
    end
end

function value = getfield_default_local(s, name, defaultValue)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
        value = s.(name);
    else
        value = defaultValue;
    end
end

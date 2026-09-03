function [predictFcn, info] = make_single_generator_dynamic_predictor(methodName, methodResult)
%MAKE_SINGLE_GENERATOR_DYNAMIC_PREDICTOR Build one fixed rollout predictor.
%
% The compiled PhDN/SINDy path is preferred. If compilation cannot support a
% legacy term, the function falls back to the original fixed-model forward
% routine instead of terminating the complete rollout script.

    rawMethodName = lower(strtrim(char(methodName)));
    methodFamily = canonical_method_family_local(rawMethodName,methodResult);
    predictFcn = [];
    info = struct('available', false, 'reason', '', 'method', rawMethodName, ...
        'methodFamily',methodFamily,'compiled', false, ...
        'compileTime', 0, 'compileError', '');

    switch methodFamily
        case 'phdn'
            if ~isfield(methodResult, 'Coef_M_est') || isempty(methodResult.Coef_M_est) || ...
                    ~isfield(methodResult, 'arch') || isempty(methodResult.arch)
                info.reason = 'PhDN coefficient/architecture fields are unavailable.';
                return;
            end

            [referenceFcn, normOpt] = make_original_phdn_predictor_local(methodResult);
            info.referencePredictFcn = referenceFcn;
            compileTimer = tic;
            try
                compiled = compile_single_generator_phdn_predictor(methodResult);
                predictFcn = @(X) predict_compiled_single_generator_phdn(compiled, X);
                info.available = true;
                info.compiled = true;
                info.compileTime = toc(compileTimer);
                info.activeTermCount = compiled.activeTermCount;
                info.fallbackTermCount = compiled.fallbackTermCount;
                info.reason = sprintf(['Compiled fixed PhDN inference: %d active ', ...
                    'term columns, %d fallback terms, no Jacobian/dictionary ', ...
                    'reconstruction during rollout (compile %.3f s).'], ...
                    compiled.activeTermCount, compiled.fallbackTermCount, ...
                    info.compileTime);
            catch ME
                predictFcn = referenceFcn;
                info.available = true;
                info.compiled = false;
                info.compileTime = toc(compileTimer);
                info.compileError = ME.message;
                info.reason = sprintf(['Compiled PhDN inference was unavailable; ', ...
                    'using the original fixed MODEL_FORWARD path. Compile error: %s'], ...
                    ME.message);
                warning('SingleGeneratorDynamic:CompiledPhDNFallback', '%s', info.reason);
            end

        case {'stage0-sr', 'stage0sr'}
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
            referenceFcn = @(X) predict_sindy_baseline(methodResult, X);
            info.referencePredictFcn = referenceFcn;
            compileTimer = tic;
            try
                compiled = compile_single_generator_sindy_predictor(methodResult);
                predictFcn = @(X) predict_compiled_single_generator_sindy(compiled, X);
                info.available = true;
                info.compiled = true;
                info.compileTime = toc(compileTimer);
                info.activeTermCount = compiled.activeTermCount;
                info.fallbackTermCount = compiled.fallbackTermCount;
                info.reason = sprintf(['Compiled fixed SINDy inference: %d active ', ...
                    'terms, %d fallback terms; no full-library reconstruction ', ...
                    'during rollout (compile %.3f s).'], ...
                    compiled.activeTermCount, compiled.fallbackTermCount, ...
                    info.compileTime);
            catch ME
                predictFcn = referenceFcn;
                info.available = true;
                info.compiled = false;
                info.compileTime = toc(compileTimer);
                info.compileError = ME.message;
                info.reason = sprintf(['Compiled SINDy inference was unavailable; ', ...
                    'using the original fixed PREDICT_SINDY_BASELINE path. ', ...
                    'Compile error: %s'], ME.message);
                warning('SingleGeneratorDynamic:CompiledSINDyFallback', '%s', info.reason);
            end

        case {'kan','kan-pruned'}
            if isfield(methodResult,'portableModel') && isstruct(methodResult.portableModel) && ...
                    ~isempty(fieldnames(methodResult.portableModel))
                predictFcn=@(X) predict_single_generator_dynamic_kan_baseline(methodResult,X);
                info.available=true; info.compiled=true;
                info.reason=['Portable validation-selected pyKAN spline model ', ...
                    'evaluated directly in MATLAB for arbitrary-state rollout.'];
            else
                info.reason='Selected KAN portable model is unavailable.';
            end

        case {'eql','eql-div'}
            if isfield(methodResult,'portableModel') && isstruct(methodResult.portableModel) && ...
                    ~isempty(fieldnames(methodResult.portableModel))
                predictFcn=@(X) predict_single_generator_dynamic_eql_baseline(methodResult,X);
                info.available=true; info.compiled=true;
                info.reason=['Portable external-validation-selected EQL-Div model ', ...
                    'evaluated directly in MATLAB for arbitrary-state rollout.'];
            else
                info.reason='Selected EQL portable model is unavailable.';
            end

        case 'mlp'
            if isfield(methodResult, 'net') && ~isempty(methodResult.net)
                predictFcn = @(X) methodResult.net(X.').';
                info.available = true;
            elseif isfield(methodResult, 'modelParameters') && ...
                    isstruct(methodResult.modelParameters) && ...
                    isfield(methodResult.modelParameters, 'weights') && ...
                    ~isempty(methodResult.modelParameters.weights)
                predictFcn = @(X) predict_single_generator_dynamic_mlp_baseline( ...
                    methodResult, X);
                info.available = true;
                info.reason = ['Arbitrary-state predictor evaluates the exported ', ...
                    'validation-selected Python MLP weights.'];
            else
                info.reason = ['No MATLAB network or exported Python MLP weights ', ...
                    'are available for arbitrary-state rollout.'];
            end

        otherwise
            info.reason = sprintf(['No arbitrary-state predictor is registered for %s. ', ...
                'Derivative/test metrics remain available.'], rawMethodName);
    end
end


function family = canonical_method_family_local(methodName,methodResult)
    family = '';
    if isstruct(methodResult) && isfield(methodResult,'methodFamily') && ...
            ~isempty(methodResult.methodFamily)
        family = lower(strtrim(char(methodResult.methodFamily)));
    end
    if isempty(family)
        compact = regexprep(lower(strtrim(char(methodName))),'[^a-z0-9]','');
        if startsWith(compact,'phdn')
            family = 'phdn';
        elseif startsWith(compact,'stage0sr') || ...
                (contains(compact,'stage0') && contains(compact,'sr'))
            family = 'stage0-sr';
        elseif startsWith(compact,'sindy')
            family = 'sindy';
        elseif startsWith(compact,'kan')
            family = 'kan';
        elseif startsWith(compact,'eql')
            family = 'eql-div';
        elseif startsWith(compact,'mlp')
            family = 'mlp';
        else
            family = lower(strtrim(char(methodName)));
        end
    end
end

function [predictFcn, normOpt] = make_original_phdn_predictor_local(methodResult)
    useIONorm = false;
    useLayerNorm = false;
    if isfield(methodResult, 'opts') && isfield(methodResult.opts, 'norm')
        useIONorm = logical(getfield_default_local( ...
            methodResult.opts.norm, 'useInputOutputNorm', false));
        useLayerNorm = logical(getfield_default_local( ...
            methodResult.opts.norm, 'useLayerNorm', false));
    end
    data = methodResult.data;
    normOpt = fit_norm_options(data.Xtr, data.Ytr, useIONorm, useLayerNorm);
    predictFcn = @(X) model_forward(X, methodResult.Coef_M_est, ...
        methodResult.arch, normOpt);
end

function value = getfield_default_local(s, name, defaultValue)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        value = s.(name);
    else
        value = defaultValue;
    end
end

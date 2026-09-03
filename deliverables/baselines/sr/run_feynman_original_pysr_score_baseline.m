function resultSR = run_feynman_original_pysr_score_baseline(task, opts, dataResult)
%RUN_FEYNMAN_ORIGINAL_PYSR_SCORE_BASELINE Independent Stage-0 PySR score ablation.
% Search grammar, data, budgets, seeds, SINDy bypass and restart machinery are
% inherited from opts.  The only scientific change is final expression
% selection: PySR's exported native score replaces the proposed PhDN score.

    if nargin < 3 || isempty(dataResult) || ~isstruct(dataResult) || ~isfield(dataResult,'data')
        error('A shared baseline/PhDN data result is required.');
    end
    nativeOpts = opts;
    nativeOpts.stage0.pysr.selectionPolicy = 'pysr_native_score';
    % This flag is also disabled for transparent reporting.  selectionPolicy
    % is the authoritative switch and bypasses the machine-floor PhDN rule.
    nativeOpts.stage0.pysr.structureScoreEnable = false;

    archBase = task.arch;
    archBase.nx = task.nx;
    archBase.ny = task.ny;
    archBase.safety = nativeOpts.safety;
    archBase.feasibility = nativeOpts.training;

    t = tic;
    stage0Result = run_phdn_per_output_pysr_stage0( ...
        task, archBase, dataResult.data, nativeOpts.stage0);
    elapsed = toc(t);

    M = getfield_default_local(stage0Result,'bestModel',struct());
    resultSR = struct();
    resultSR.available = true;
    resultSR.method = 'PySR-original-score';
    resultSR.reportRole = 'feynman_original_pysr_score_ablation';
    resultSR.reportTitle = 'PySR baseline with original/native score selection';
    resultSR.selectionPolicy = 'pysr_native_score';
    resultSR.stage0Result = stage0Result;
    resultSR.usedSingleLayerBypass = logical(getfield_default_local(stage0Result,'usedSingleLayerBypass',false));
    resultSR.usedPerOutputSindyBypass = logical(getfield_default_local(stage0Result,'usedPerOutputSindyBypass',false));
    resultSR.bypassOutputMask = getfield_default_local(stage0Result,'bypassOutputMask',[]);
    resultSR.searchExecuted = ~resultSR.usedSingleLayerBypass;
    resultSR.bypassed = resultSR.usedSingleLayerBypass;
    resultSR.status = ternary_local(resultSR.bypassed,'BYPASS','AVAILABLE');
    resultSR.reason = ternary_local(resultSR.bypassed, ...
        'all_outputs_used_stage0_sindy_bypass_no_pysr_search', ...
        'independent_pysr_search_selected_by_native_score');
    resultSR.selectedExpressions = getfield_default_local(stage0Result,'bestExpressions', ...
        getfield_default_local(M,'outputExpressions',{}));
    resultSR.structureLabel = format_expression_label_local(resultSR.selectedExpressions);
    resultSR.outputSelections = getfield_default_local(M,'outputSelections',struct([]));
    resultSR.valMetrics = getfield_default_local(M,'valMetrics',struct('mse',NaN,'rmse',NaN));
    resultSR.testMetrics = getfield_default_local(M,'testMetrics',struct('mse',NaN,'rmse',NaN));
    resultSR.oodMetrics = getfield_default_local(M,'oodMetrics',struct('mse',NaN,'rmse',NaN));
    resultSR.trainMetrics = getfield_default_local(M,'trainMetrics',struct());
    resultSR.trainTime = getfield_default_local(stage0Result,'trainTime',elapsed);
    resultSR.stage0SearchTime = getfield_default_local(stage0Result,'searchTime',NaN);
    resultSR.stage0BaseDictionaryTime = getfield_default_local(stage0Result,'baseDictionaryTime',NaN);
    resultSR.timeStats = struct('total',resultSR.trainTime,'stage0Total',resultSR.trainTime, ...
        'pysrSearch',resultSR.stage0SearchTime,'baseDictionary',resultSR.stage0BaseDictionaryTime);
    resultSR.nActiveCoefficients = getfield_default_local(M,'nActiveTerms', ...
        getfield_default_local(M,'nActiveCoefficients',getfield_default_local(M,'complexity',NaN)));
    resultSR.complexity = getfield_default_local(M,'complexity',NaN);

    fprintf('\n========================================\n');
    fprintf('Feynman PySR original-score baseline\n');
    fprintf('========================================\n');
    fprintf('Selection policy : %s\n', resultSR.selectionPolicy);
    fprintf('Status           : %s\n', resultSR.status);
    if ~isempty(resultSR.selectedExpressions)
        fprintf('Selected expr.   : %s\n', resultSR.structureLabel);
    end
    fprintf('Validation RMSE  : %.6e\n', metric_local(resultSR.valMetrics,'rmse'));
    fprintf('ID-test RMSE     : %.6e\n', metric_local(resultSR.testMetrics,'rmse'));
    fprintf('OOD RMSE         : %.6e\n', metric_local(resultSR.oodMetrics,'rmse'));
    fprintf('Stage-0 time [s] : %.3f\n', resultSR.trainTime);
end

function x=metric_local(S,n); x=getfield_default_local(S,n,NaN); end
function v=getfield_default_local(S,n,d); if isstruct(S)&&isfield(S,n)&&~isempty(S.(n)); v=S.(n); else; v=d; end; end
function y=ternary_local(c,a,b); if c; y=a; else; y=b; end; end
function label=format_expression_label_local(e)
    if isempty(e); label=''; return; end
    if ischar(e)||isstring(e); e=cellstr(e); end
    p=cell(1,numel(e)); for i=1:numel(e); p{i}=sprintf('y%d=%s',i,char(string(e{i}))); end
    label=strjoin(p,'; ');
end
end

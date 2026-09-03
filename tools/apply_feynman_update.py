from pathlib import Path
import re, shutil, zipfile, copy, xml.etree.ElementTree as ET, difflib, json

ROOT=Path('.')
SRC=ROOT/'unpacked'
OUT=ROOT/'deliverables'
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir(parents=True)

def read(rel): return (SRC/rel).read_text(encoding='utf-8')
def write(rel,s):
    p=OUT/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(s,encoding='utf-8',newline='\n')

def replace_once(s, old, new, label):
    if old not in s: raise RuntimeError(f'anchor not found: {label}')
    if s.count(old)!=1: raise RuntimeError(f'anchor not unique ({s.count(old)}): {label}')
    return s.replace(old,new,1)

# ------------------------------------------------------------------
# Defaults: explicit selection policy, preserving proposed PhDN default.
# ------------------------------------------------------------------
s=read('core/phdnn_default_options.m')
s=replace_once(s,"\topts.stage0.pysr.modelSelection = 'best';\n", "\topts.stage0.pysr.modelSelection = 'best';\n\t% Final candidate selector after PySR search/restart pooling.\n\t% 'phdn_structure_score' keeps the proposed PhDN rule;\n\t% 'pysr_native_score' is the clean original-PySR-score ablation.\n\topts.stage0.pysr.selectionPolicy = 'phdn_structure_score';\n",'phdnn default selection policy')
write('core/phdnn_default_options.m',s)

s=read('baselines/sr/sr_default_options.m')
s=replace_once(s,"    opts.modelSelection = 'best';\n", "    opts.modelSelection = 'best';\n    % Wrapper-level final selector. Official standalone SR defaults to native\n    % PySR score; PhDN explicitly overrides this to phdn_structure_score.\n    opts.selectionPolicy = 'pysr_native_score';\n",'sr default selection policy')
write('baselines/sr/sr_default_options.m',s)

# ------------------------------------------------------------------
# MATLAB -> Python adapter config.
# ------------------------------------------------------------------
s=read('baselines/sr/train_official_pysr_baseline.m')
s=replace_once(s,"    cfg.structure_score_enable = logical(getfield_default_local(srOpts, 'structureScoreEnable', false));\n", "    cfg.selection_policy = char(string(getfield_default_local(srOpts, 'selectionPolicy', 'pysr_native_score')));\n    cfg.structure_score_enable = logical(getfield_default_local(srOpts, 'structureScoreEnable', false));\n",'train adapter selection policy')
write('baselines/sr/train_official_pysr_baseline.m',s)

# ------------------------------------------------------------------
# Python per-run selector: native policy bypasses PhDN machine-floor rule.
# ------------------------------------------------------------------
s=read('baselines/sr/pysr_official_adapter.py')
anchor="""    if not records:\n        raise ValueError('Cannot select a Stage-0 core from an empty record list.')\n\n    machine_floor_indices = [\n"""
insert="""    if not records:\n        raise ValueError('Cannot select a Stage-0 core from an empty record list.')\n\n    selection_policy = str(config.get('selection_policy', 'pysr_native_score')).strip().lower()\n    native_score_policy = selection_policy in {\n        'pysr_native_score', 'native_pysr_score', 'original_pysr_score', 'original_score'\n    }\n    if native_score_policy:\n        # Clean ablation: use PySR's exported native score directly.  Do not\n        # apply the PhDN numerical-floor simplicity rule, external-validation\n        # composite score, or structure-frontier score before this decision.\n        finite_score_indices: List[int] = []\n        for i, rec in enumerate(records):\n            try:\n                score = float(rec.get('score', np.nan))\n            except Exception:\n                score = np.nan\n            if np.isfinite(score):\n                finite_score_indices.append(i)\n        if finite_score_indices:\n            core_idx = min(\n                finite_score_indices,\n                key=lambda i: (-float(records[i]['score']),\n                               float(records[i].get('validation_mse', float('inf'))),\n                               float(records[i].get('complexity', float('inf')))))\n            return core_idx, 'maximum_native_pysr_score'\n        core_idx = min(\n            range(len(records)),\n            key=lambda i: (float(records[i].get('validation_mse', float('inf'))),\n                           float(records[i].get('complexity', float('inf')))))\n        return core_idx, 'external_validation_fallback_no_finite_native_score'\n\n    machine_floor_indices = [\n"""
s=replace_once(s,anchor,insert,'python native policy')
write('baselines/sr/pysr_official_adapter.py',s)

# ------------------------------------------------------------------
# Stage-0 wrapper: forward policy and make cross-restart native-score choice.
# ------------------------------------------------------------------
s=read('core/stage0/run_phdn_per_output_pysr_stage0.m')
s=replace_once(s,"    srOpts.modelSelection = options.pysr.modelSelection;\n", "    srOpts.modelSelection = options.pysr.modelSelection;\n    srOpts.selectionPolicy = char(string(getfield_default_local( ...\n        options.pysr, 'selectionPolicy', 'phdn_structure_score')));\n",'stage0 forward policy')
s=replace_once(s,"    useStructureScore = logical(getfield_default_local(pysrOptions, 'structureScoreEnable', true));\n", "    selectionPolicy = lower(strtrim(char(string(getfield_default_local( ...\n        pysrOptions, 'selectionPolicy', 'phdn_structure_score')))));\n    useNativePySRScore = any(strcmp(selectionPolicy, ...\n        {'pysr_native_score','native_pysr_score','original_pysr_score','original_score'}));\n    useStructureScore = logical(getfield_default_local(pysrOptions, 'structureScoreEnable', true)) && ...\n        ~useNativePySRScore;\n",'merge policy')
s=replace_once(s,"        floorCandidates = find(validationValues <= selectionMSEFloor);\n        usedMachineFloorTie = ~isempty(floorCandidates);\n        if usedMachineFloorTie\n", "        floorCandidates = find(validationValues <= selectionMSEFloor);\n        usedMachineFloorTie = ~useNativePySRScore && ~isempty(floorCandidates);\n        if useNativePySRScore\n            % Clean original-score ablation across all exported restart\n            % candidates.  Native score is the primary and only scientific\n            % selector; validation/complexity are deterministic exact-score\n            % tie breakers only.  The proposed numerical-floor simplicity\n            % rule and structure score are deliberately bypassed.\n            nativeScores = NaN(1,numel(pool));\n            nativeVals = Inf(1,numel(pool));\n            nativeComplexities = Inf(1,numel(pool));\n            for qNative = 1:numel(pool)\n                nativeScores(qNative) = getfield_default_local(pool{qNative},'score',NaN);\n                nativeVals(qNative) = scalar_or_inf_local(getfield_default_local(pool{qNative},'validation_mse',Inf));\n                nativeComplexities(qNative) = scalar_or_inf_local(getfield_default_local(pool{qNative},'complexity',Inf));\n            end\n            nativeCandidates = find(isfinite(nativeScores));\n            if isempty(nativeCandidates)\n                [~,nativeOrder] = sortrows([nativeVals(:),nativeComplexities(:)],[1 2]);\n                selectedPoolIndex = nativeOrder(1);\n                nativeRule = 'external_validation_fallback_no_finite_native_score';\n            else\n                [~,nativeOrder] = sortrows([-nativeScores(nativeCandidates).', ...\n                    nativeVals(nativeCandidates).',nativeComplexities(nativeCandidates).'],[1 2 3]);\n                selectedPoolIndex = nativeCandidates(nativeOrder(1));\n                nativeRule = 'maximum_native_pysr_score_cross_restart';\n            end\n            selectedCandidate = pool{selectedPoolIndex};\n            selectedCandidate.selection_rule = nativeRule;\n            selectedCandidate.selection_mse_floor = selectionMSEFloor;\n            selectedCandidate.validation_floor_tied = false;\n            selectedCandidate.selection_prefers_simplicity_within_floor = false;\n            selectedCandidate.selection_role = 'final-structure-core';\n            selectedCandidate.ranking_scope = 'cross_restart_global_native_pysr_score';\n            selectedCandidate.relative_error_scope = 'cross_restart_global_validation_best';\n        elseif usedMachineFloorTie\n",'cross restart native branch')
s=replace_once(s,"        if usedMachineFloorTie\n            selectedSelection.selectionMode = 'machine_floor_simplicity_tie';\n        elseif useStructureScore\n", "        if useNativePySRScore\n            selectedSelection.selectionMode = char(string(getfield_default_local( ...\n                selectedCandidate,'selection_rule','maximum_native_pysr_score_cross_restart')));\n        elseif usedMachineFloorTie\n            selectedSelection.selectionMode = 'machine_floor_simplicity_tie';\n        elseif useStructureScore\n",'selection mode')
write('core/stage0/run_phdn_per_output_pysr_stage0.m',s)

# ------------------------------------------------------------------
# Independent Feynman original-PySR-score baseline.
# ------------------------------------------------------------------
write('baselines/sr/run_feynman_original_pysr_score_baseline.m',r'''function resultSR = run_feynman_original_pysr_score_baseline(task, opts, dataResult)
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
''')

# ------------------------------------------------------------------
# Feynman persistence/replay helpers.
# ------------------------------------------------------------------
write('baselines/common/save_feynman_dimless_method_result.m',r'''function save_feynman_dimless_method_result(resultsFile,iRound,caseName,methodField,methodLabel,result,context)
%SAVE_FEYNMAN_DIMLESS_METHOD_RESULT Immediate non-destructive method checkpoint.
    if nargin<7 || isempty(context); context=struct(); end
    folder=fileparts(resultsFile); if ~exist(folder,'dir'); mkdir(folder); end
    DB=struct();
    if exist(resultsFile,'file')
        S=load(resultsFile,'feynmanResults'); if isfield(S,'feynmanResults'); DB=S.feynmanResults; end
    end
    rkey=sprintf('round_%02d',iRound); ckey=matlab.lang.makeValidName(strrep(caseName,'.','p'));
    if ~isfield(DB,rkey); DB.(rkey)=struct(); end
    if ~isfield(DB.(rkey),ckey); DB.(rkey).(ckey)=struct(); end
    if ~isfield(DB.(rkey).(ckey),'methods'); DB.(rkey).(ckey).methods=struct(); end
    rec=struct('methodField',methodField,'methodLabel',methodLabel,'round',iRound, ...
        'caseName',caseName,'savedAt',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')), ...
        'result',result,'context',context);
    DB.(rkey).(ckey).methods.(methodField)=rec;
    DB.schema='feynman_dimless_independent_method_record_v1';
    DB.lastUpdated=rec.savedAt;
    feynmanResults=DB; %#ok<NASGU>
    save(resultsFile,'feynmanResults','-v7.3');
    fprintf('[Feynman checkpoint] %s | round %d | %s -> %s\n',caseName,iRound,methodLabel,resultsFile);
end
''')

write('baselines/common/load_feynman_dimless_method_result.m',r'''function [result,found,record] = load_feynman_dimless_method_result(resultsFile,iRound,caseName,methodField,methodLabel,compact)
%LOAD_FEYNMAN_DIMLESS_METHOD_RESULT Load one method without touching other records.
    if nargin<6; compact=true; end
    result=[]; found=false; record=struct();
    if ~exist(resultsFile,'file')
        warning('Feynman results file not found: %s',resultsFile); return;
    end
    S=load(resultsFile,'feynmanResults'); if ~isfield(S,'feynmanResults'); return; end
    rkey=sprintf('round_%02d',iRound); ckey=matlab.lang.makeValidName(strrep(caseName,'.','p'));
    try
        record=S.feynmanResults.(rkey).(ckey).methods.(methodField);
        result=record.result; found=true;
    catch
        warning('No saved Feynman record: case=%s round=%d method=%s.',caseName,iRound,methodLabel); return;
    end
    fprintf('\n[Feynman replay] %s | round %d | %s\n',caseName,iRound,methodLabel);
    if isfield(record,'savedAt'); fprintf('Saved at          : %s\n',record.savedAt); end
    if ~compact; disp(result); return; end
    expr=getfield_default_local(result,'selectedExpressions',{});
    if isempty(expr) && isfield(result,'stage0Expressions'); expr=result.stage0Expressions; end
    if ~isempty(expr)
        if ischar(expr)||isstring(expr); expr=cellstr(expr); end
        parts=cell(1,numel(expr)); for k=1:numel(expr); parts{k}=sprintf('y%d=%s',k,char(string(expr{k}))); end
        fprintf('Selected expr.    : %s\n',strjoin(parts,'; '));
    end
    fprintf('Validation RMSE   : %.6e\n',metric_from_result_local(result,'valMetrics','rmse'));
    fprintf('ID-test RMSE      : %.6e\n',metric_from_result_local(result,'testMetrics','rmse'));
    fprintf('OOD RMSE          : %.6e\n',metric_from_result_local(result,'oodMetrics','rmse'));
    t=getfield_default_local(result,'trainTime',NaN);
    if ~isfinite(t) && isfield(result,'timeStats'); t=getfield_default_local(result.timeStats,'total',NaN); end
    fprintf('Training time [s] : %.3f\n',t);
end
function x=metric_from_result_local(R,f,n)
    x=NaN; if isstruct(R)&&isfield(R,f)&&isstruct(R.(f)); x=getfield_default_local(R.(f),n,NaN); end
    if ~isfinite(x)&&strcmp(f,'testMetrics')&&isfield(R,'metrics'); x=getfield_default_local(R.metrics,'rmse',NaN); end
end
function v=getfield_default_local(S,n,d); if isstruct(S)&&isfield(S,n)&&~isempty(S.(n)); v=S.(n); else; v=d; end; end
''')

write('baselines/common/save_feynman_dimless_summary_outputs.m',r'''function save_feynman_dimless_summary_outputs(resultsFile,allResults,summaryRows,metadata)
%SAVE_FEYNMAN_DIMLESS_SUMMARY_OUTPUTS Merge aggregate report without erasing method checkpoints.
    if nargin<4; metadata=struct(); end
    DB=struct();
    if exist(resultsFile,'file')
        S=load(resultsFile,'feynmanResults'); if isfield(S,'feynmanResults'); DB=S.feynmanResults; end
    end
    DB.aggregate=allResults;
    DB.summaryRows=summaryRows;
    DB.metadata=metadata;
    DB.schema='feynman_dimless_independent_method_record_v1';
    DB.lastUpdated=char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    feynmanResults=DB; %#ok<NASGU>
    save(resultsFile,'feynmanResults','-v7.3');
    fprintf('[Feynman summary] merged aggregate results -> %s\n',resultsFile);
end
''')

# ------------------------------------------------------------------
# Main Feynman script patch.
# ------------------------------------------------------------------
main=(ROOT/'mlx_code/run_demo_feynman_dimless.m').read_text(encoding='utf-8')
main=main.replace('% already-completed Stage-0 search; it has no separate switch, options, or rerun.',
                  '% already-completed Stage-0 search. An independent PySR-original-score ablation is also available.')
main=replace_once(main,"addpath(genpath(projectRoot));\nrehash;\n", "addpath(genpath(projectRoot));\nrehash;\n\n% -------------------------------------------------------------------------\n% Feynman result persistence / replay\n% -------------------------------------------------------------------------\nOutputCaseRoot = fullfile(projectRoot,'outputs','Feynman_Dimless');\nif ~exist(OutputCaseRoot,'dir'); mkdir(OutputCaseRoot); end\nResultsFile = fullfile(OutputCaseRoot,'feynman_dimless_results.mat');\nSaveResults = true;\nRecordedReportCompactMode = true;\n",'main persistence path')
old="""RunPhDNMainModel = true;\nRunMLPBaseline = true;\nRunEQLBaseline = true;\nRunKANBaseline = true;\nRunSINDyBaseline = true;\n"""
new="""RunPhDNMainModel = true;\nRunOriginalPySRScoreBaseline = true;  % independent PySR search; native/original score selects expression\nRunMLPBaseline = true;\nRunEQLBaseline = true;\nRunKANBaseline = true;\nRunSINDyBaseline = true;\n\n% Replay controls. Run*=true always takes priority over the matching replay flag.\ndisplayRecordReport_PhDN = false;\ndisplayRecordReport_Stage0SR = false;\ndisplayRecordReport_PySROriginalScore = false;\ndisplayRecordReport_MLP = false;\ndisplayRecordReport_EQL = false;\ndisplayRecordReport_KAN = false;\ndisplayRecordReport_SINDy = false;\n"""
main=replace_once(main,old,new,'method switches')
main=replace_once(main,"    opts.stage0.pysr.modelSelection = Stage0ModelSelection;\n", "    opts.stage0.pysr.modelSelection = Stage0ModelSelection;\n    opts.stage0.pysr.selectionPolicy = 'phdn_structure_score'; % proposed PhDN selector\n",'main proposed policy')
start=main.index('    runAnyBaseline = RunMLPBaseline || RunEQLBaseline || RunKANBaseline || RunSINDyBaseline;')
end=main.index('    % Refresh the stored PhDN object after attaching baseline and ablation records.')
prefix=main[:start]; suffix=main[end:]
block=r'''    runAnyBaseline = RunOriginalPySRScoreBaseline || RunMLPBaseline || RunEQLBaseline || RunKANBaseline || RunSINDyBaseline;
    runAnyMethod = RunPhDNMainModel || runAnyBaseline;
    replayAnyMethod = displayRecordReport_PhDN || displayRecordReport_Stage0SR || ...
        displayRecordReport_PySROriginalScore || displayRecordReport_MLP || ...
        displayRecordReport_EQL || displayRecordReport_KAN || displayRecordReport_SINDy;
    result = [];
    resultStage0SR = [];
    baselineDataResult = [];
    resultPack = struct();
    context = struct('caseName',task.name,'round',iRound,'casemode',casemode, ...
        'stage0RandomState',roundStage0RandomState,'matlabDataSeed',1, ...
        'selectionPolicyProposed','phdn_structure_score', ...
        'selectionPolicyOriginalPySR','pysr_native_score');

    % ---------------- PhDN + collected proposed Stage-0 SR ----------------
    if RunPhDNMainModel
        result = phdnn_identify(task, opts);
        print_demo_output(task, result);
        if isfield(result,'ablations') && isstruct(result.ablations) && isfield(result.ablations,'stage0SR')
            resultStage0SR = result.ablations.stage0SR;
        else
            resultStage0SR = make_stage0_sr_ablation_result(result);
            result.ablations.stage0SR = resultStage0SR;
        end
        print_stage0_sr_ablation_result(resultStage0SR);
        resultPack.phdn = result;
        resultPack.stage0sr = resultStage0SR;
        baselineDataResult = result;
        if SaveResults
            save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'phdn','PhDN',result,context);
            save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'stage0sr','Stage0-SR (proposed score)',resultStage0SR,context);
        end
    else
        if displayRecordReport_PhDN
            [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'phdn','PhDN',RecordedReportCompactMode);
            if ok; resultPack.phdn=r; end
        end
        if displayRecordReport_Stage0SR
            [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'stage0sr','Stage0-SR (proposed score)',RecordedReportCompactMode);
            if ok; resultPack.stage0sr=r; end
        end
    end

    % Generate exactly one shared split only when a currently-run independent
    % baseline needs it and PhDN did not already provide that split.
    if runAnyBaseline && isempty(baselineDataResult)
        baselineDataResult = make_baseline_data_result_from_task(task, opts);
        fprintf('PhDN main model skipped; generated one shared data split for enabled baselines.\n');
    end

    % ---------------- Original/native PySR-score ablation ----------------
    if RunOriginalPySRScoreBaseline
        nativeOpts = opts;
        nativeOpts.stage0.pysr.selectionPolicy = 'pysr_native_score';
        nativeOpts.stage0.pysr.structureScoreEnable = false;
        nativeOpts.stage0.pysr.workRoot = fullfile(Stage0WorkRoot,'original_pysr_score');
        resultPySROriginal = run_feynman_original_pysr_score_baseline(task,nativeOpts,baselineDataResult);
        resultPack.pysrOriginalScore = resultPySROriginal;
        if SaveResults
            save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'pysrOriginalScore','PySR-original-score',resultPySROriginal,context);
        end
    elseif displayRecordReport_PySROriginalScore
        [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'pysrOriginalScore','PySR-original-score',RecordedReportCompactMode);
        if ok; resultPack.pysrOriginalScore=r; end
    end

    % ---------------- MLP ----------------
    if RunMLPBaseline
        mlpOpts = make_default_mlp_options_for_demo(roundMlpSeed);
        mlpOpts.protocol = MLPProtocol; mlpOpts.pythonExe = MLPPythonExe; mlpOpts.pykanRoot = PyKANRoot;
        mlpOpts.workRoot = MLPWorkRoot; mlpOpts.width = MLPWidth; mlpOpts.depthList = MLPDepthList;
        mlpOpts.activationList = MLPActivationList; mlpOpts.optimizer = MLPOptimizer; mlpOpts.steps = MLPTrainSteps;
        mlpOpts.learningRate = MLPLearningRate; mlpOpts.verbose = MLPVerbose; mlpOpts.displaySweepTable = MLPDisplaySweepTable;
        resultMlp = run_mlp_baseline_from_phdn_result(baselineDataResult, mlpOpts); resultPack.mlp = resultMlp;
        if RunPhDNMainModel; result.baselines.mlp = resultMlp; end
        if SaveResults; save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'mlp','MLP',resultMlp,context); end
    elseif displayRecordReport_MLP
        [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'mlp','MLP',RecordedReportCompactMode); if ok; resultPack.mlp=r; end
    end

    % ---------------- EQL-Div ----------------
    if RunEQLBaseline
        eqlOpts = make_default_eql_options_for_demo(roundEqlSeed);
        eqlOpts.pythonExe=EQLPythonExe; eqlOpts.officialRoot=EQLOfficialRoot; eqlOpts.workRoot=EQLWorkRoot;
        eqlOpts.depthList=EQLDepthList; eqlOpts.lambdaList=EQLLambdaList; eqlOpts.unitsPerUnaryType=EQLUnitsPerUnaryType;
        eqlOpts.multiplicationUnits=EQLUnitsPerUnaryType; eqlOpts.stepsPerHiddenLayer=EQLStepsPerHiddenLayer;
        eqlOpts.batchSize=EQLBatchSize; eqlOpts.learningRate=EQLLearningRate; eqlOpts.gradient=EQLGradient;
        eqlOpts.lambdaL2=EQLLambdaL2; eqlOpts.penaltyEvery=EQLPenaltyEvery; eqlOpts.validateEvery=EQLValidateEvery;
        eqlOpts.candidateWorkers=EQLCandidateWorkers; eqlOpts.normalizeInputs=EQLNormalizeInputs; eqlOpts.normalizeOutputs=EQLNormalizeOutputs;
        eqlOpts.theanoFlags=EQLTheanoFlags; eqlOpts.verbose=EQLVerbose; eqlOpts.displaySweepTable=EQLDisplaySweepTable;
        resultEql=run_eql_baseline_from_phdn_result(baselineDataResult,eqlOpts); resultPack.eql=resultEql;
        if RunPhDNMainModel; result.baselines.eql=resultEql; end
        if SaveResults; save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'eql','EQL-Div',resultEql,context); end
    elseif displayRecordReport_EQL
        [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'eql','EQL-Div',RecordedReportCompactMode); if ok; resultPack.eql=r; end
    end

    % ---------------- KAN ----------------
    if RunKANBaseline
        kanOpts=make_default_kan_options_for_demo(roundKanSeed);
        kanOpts.pythonExe=KANPythonExe; kanOpts.pykanRoot=PyKANRoot; kanOpts.workRoot=KANWorkRoot;
        kanOpts.width=KANWidth; kanOpts.depthList=KANDepthList; kanOpts.gridList=KANGridList; kanOpts.splineOrder=KANSplineOrder;
        kanOpts.sparsificationLambdaList=KANSparsificationLambdaList; kanOpts.stepsPerGrid=KANStepsPerGrid;
        kanOpts.optimizer=KANOptimizer; kanOpts.learningRate=KANLearningRate; kanOpts.pruneNodeThreshold=KANPruneNodeThreshold;
        kanOpts.pruneEdgeThreshold=KANPruneEdgeThreshold; kanOpts.verbose=KANVerbose; kanOpts.displaySweepTable=KANDisplaySweepTable;
        resultKan=run_kan_baseline_from_phdn_result(baselineDataResult,kanOpts); resultPack.kan=resultKan;
        if RunPhDNMainModel; result.baselines.kan=resultKan; end
        if SaveResults; save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'kan','KAN-pruned',resultKan,context); end
    elseif displayRecordReport_KAN
        [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'kan','KAN-pruned',RecordedReportCompactMode); if ok; resultPack.kan=r; end
    end

    % ---------------- SINDy ----------------
    if RunSINDyBaseline
        sindyOpts=make_default_sindy_options_for_demo(); sindyOpts.thresholdList=SINDyThresholdList;
        sindyOpts.maxSTLSQIter=SINDyMaxSTLSQIter; sindyOpts.ridgeLambda=SINDyRidgeLambda; sindyOpts.dictionaryMode=SINDyDictionaryMode;
        sindyOpts.polyOrder=SINDyPolyOrder; sindyOpts.unaryOperators=SINDyUnaryOperators; sindyOpts.includeUnaryOnMonomials=SINDyIncludeUnaryOnMonomials;
        sindyOpts.includeOperatorCrossTerms=SINDyIncludeOperatorCrossTerms; sindyOpts.usePhdnDictionarySupport=SINDyUsePhdnDictionarySupport;
        sindyOpts.centerScaleLibrary=SINDyCenterScaleLibrary; sindyOpts.removeNearConstantRows=SINDyRemoveNearConstantRows;
        sindyOpts.verbose=SINDyVerbose; sindyOpts.maxTermsToPrint=SINDyMaxTermsToPrint;
        resultSindy=run_sindy_baseline_from_phdn_result(baselineDataResult,task,sindyOpts,opts); resultPack.sindy=resultSindy;
        if RunPhDNMainModel; result.baselines.sindy=resultSindy; end
        if SaveResults; save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'sindy','SINDy',resultSindy,context); end
    elseif displayRecordReport_SINDy
        [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'sindy','SINDy',RecordedReportCompactMode); if ok; resultPack.sindy=r; end
    end

    if ~runAnyMethod && ~replayAnyMethod
        fprintf('No run or replay method is enabled; skip this case.\n');
        continue;
    end

'''
main=prefix+block+suffix
main=replace_once(main,"        if isfield(resultPack, 'mlp')\n", "        if isfield(resultPack, 'pysrOriginalScore')\n            [summaryRows, iSummary] = append_method_summary_row(summaryRows, iSummary, task.name, 'PySR-original-score', resultPack.pysrOriginalScore, iRound);\n        end\n        if isfield(resultPack, 'mlp')\n",'native summary row')
main=replace_once(main,"print_round_statistics_summary(summaryRows);\n", "print_round_statistics_summary(summaryRows);\n\nif SaveResults\n    meta = struct('NumRounds',NumRounds,'selectedCases',{selectedCases},'casemode',casemode, ...\n        'methodPersistenceSchema','independent_method_record_v1', ...\n        'policy','per-method immediate checkpoint + non-destructive aggregate merge');\n    save_feynman_dimless_summary_outputs(ResultsFile,allResults,summaryRows,meta);\nend\n",'summary save')
write('run_demo_feynman_dimless.m',main)

# ------------------------------------------------------------------
# Rebuild MLX by applying line-level diff to code paragraphs in document.xml.
# This retains all non-code Live Editor content/resources.
# ------------------------------------------------------------------
orig_mlx=SRC/'run_demo_feynman_dimless.mlx'
work=ROOT/'_mlx_work'
if work.exists(): shutil.rmtree(work)
work.mkdir()
with zipfile.ZipFile(orig_mlx,'r') as z: z.extractall(work)
doc=work/'matlab/document.xml'
ET.register_namespace('w','http://schemas.openxmlformats.org/wordprocessingml/2006/main')
ET.register_namespace('m','http://schemas.openxmlformats.org/officeDocument/2006/math')
tree=ET.parse(doc); root=tree.getroot()
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
XML='http://www.w3.org/XML/1998/namespace'
parent={c:p for p in root.iter() for c in p}
def is_code(p):
    st=p.find(f'./{{{W}}}pPr/{{{W}}}pStyle')
    return st is not None and st.attrib.get(f'{{{W}}}val')=='code'
def ptext(p): return ''.join((t.text or '') for t in p.findall(f'.//{{{W}}}t'))
def settext(p,text):
    ppr=p.find(f'./{{{W}}}pPr')
    for c in list(p):
        if c is not ppr: p.remove(c)
    r=ET.SubElement(p,f'{{{W}}}r'); t=ET.SubElement(r,f'{{{W}}}t'); t.set(f'{{{XML}}}space','preserve'); t.text=text
paras=[p for p in root.iter(f'{{{W}}}p') if is_code(p)]
old=[ptext(p) for p in paras]
old_file=(ROOT/'mlx_code/run_demo_feynman_dimless.m').read_text(encoding='utf-8').splitlines()
new_file=main.splitlines()
# Verify extracted code corresponds to XML code paragraphs modulo empty trailing lines.
while old and old[-1]=='': old.pop(); paras.pop()
while old_file and old_file[-1]=='': old_file.pop()
if old != old_file:
    # Robust fallback: use XML sequence itself as diff base; generated main was
    # derived from the extracted code and should still align strongly.
    ratio=difflib.SequenceMatcher(a=old_file,b=old).ratio()
    if ratio < 0.95: raise RuntimeError(f'MLX code/XML alignment too low: {ratio}')
    base=old
else:
    base=old_file
sm=difflib.SequenceMatcher(a=base,b=new_file,autojunk=False)
ops=sm.get_opcodes()
for tag,i1,i2,j1,j2 in reversed(ops):
    if tag=='equal': continue
    old_elems=paras[i1:i2]; new_lines=new_file[j1:j2]
    if old_elems:
        par=parent[old_elems[0]]; idx= list(par).index(old_elems[0]); template=old_elems[0]
        for e in old_elems: par.remove(e)
        for line in reversed(new_lines):
            q=copy.deepcopy(template); settext(q,line); par.insert(idx,q)
    else:
        # Pure insertion: insert before next code paragraph where possible.
        if i1 < len(paras):
            anchor=paras[i1]; par=parent[anchor]; idx=list(par).index(anchor); template=anchor
        else:
            anchor=paras[-1]; par=parent[anchor]; idx=list(par).index(anchor)+1; template=anchor
        for line in reversed(new_lines):
            q=copy.deepcopy(template); settext(q,line); par.insert(idx,q)
tree.write(doc,encoding='UTF-8',xml_declaration=True)
out_mlx=OUT/'run_demo_feynman_dimless.mlx'
with zipfile.ZipFile(out_mlx,'w',zipfile.ZIP_DEFLATED) as z:
    for p in work.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(work))
shutil.rmtree(work)

# README
files=[str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file()]
(OUT/'README_Feynman_update.txt').write_text('''Feynman original-PySR-score ablation + persistence/replay update\n\nKey controls in run_demo_feynman_dimless.mlx:\n  RunPhDNMainModel\n  RunOriginalPySRScoreBaseline\n  RunMLPBaseline / RunEQLBaseline / RunKANBaseline / RunSINDyBaseline\n  displayRecordReport_PhDN\n  displayRecordReport_Stage0SR\n  displayRecordReport_PySROriginalScore\n  displayRecordReport_MLP / EQL / KAN / SINDy\n  SaveResults\n\nResults file:\n  outputs/Feynman_Dimless/feynman_dimless_results.mat\n\nRun=true takes priority. If Run=false and matching displayRecordReport=true, the saved current (round,case,method) record is loaded, printed, and included in summaries. Each completed method is checkpointed immediately; the final aggregate merge does not erase other method records.\n\nThe original-PySR ablation uses selectionPolicy=pysr_native_score at BOTH the Python per-run selector and MATLAB cross-restart merger. It bypasses the proposed machine-floor simplicity and structure/validation composite selection. Search grammar/budget/seeds/data/SINDy bypass remain matched.\n''',encoding='utf-8')
print('DELIVERABLES')
for f in files: print(f)

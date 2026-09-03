function resultEql=run_single_generator_dynamic_eql_baseline_from_phdn_result(resultPhdn,eqlOpts)
%RUN_SINGLE_GENERATOR_DYNAMIC_EQL_BASELINE_FROM_PHDN_RESULT Case-local EQL sweep.
%
% Paper sample-efficiency protocol
% --------------------------------
% The reported model at N is always trained on the current N-sample data set.
% A previous smaller-N model is used only as:
%   1) a fixed-validation optimization target;
%   2) an optional initialization for an additional official EQL run on all
%      current-N samples; and
%   3) a separately labelled cumulative-envelope diagnostic.
% It is never copied unchanged into the current-N paper curve.
%
% When the first exact-N sweep does not strictly beat the previous validation
% reference, the Python adapter runs targeted independent rescue restarts for
% the best current-N configurations. This makes a monotone trend substantially
% more likely without silently relabelling a smaller-N model as an N-sample one.

    if nargin<2||isempty(eqlOpts);eqlOpts=eql_default_options();end
    d=validate_shared_data_local(resultPhdn);[XOod,YOod]=get_ood_local(d);

    [previousInfo,eqlOpts]=prepare_previous_reference_local( ...
        eqlOpts,d,XOod,YOod,size(d.Xtr,1));

    freshResult=train_single_generator_dynamic_eql_sweep_baseline( ...
        d.Xtr,d.Ytr,d.Xval,d.Yval,d.Xte,d.Yte,XOod,YOod,eqlOpts);

    resultEql=attach_exact_n_protocol_local( ...
        freshResult,previousInfo,size(d.Xtr,1));
    resultEql.dataSource='Exact PhDN/baseline raw data split';
    resultEql.data=struct('nTrain',size(d.Xtr,1),'nVal',size(d.Xval,1), ...
        'nTest',size(d.Xte,1),'nOod',size(XOod,1));
    if isfield(d,'oodDomain');resultEql.data.oodDomain=d.oodDomain;end

    if resultEql.previousCandidateAvailable && ...
            ~logical_scalar_local(resultEql.strictCurrentNImprovementAchieved,false)
        warning(['EQL exact-N adaptive search did not strictly beat the previous ', ...
            'fixed-validation reference within the configured rescue budget. ', ...
            'The current-N model is still reported honestly; the cumulative ', ...
            'envelope is stored only as a separate diagnostic.']);
    end

    if getfield_default_local(eqlOpts,'verbose',true)
        print_single_generator_dynamic_eql_result(resultEql);
    end
end

function d=validate_shared_data_local(r)
    if ~isfield(r,'data')||isempty(r.data);error('resultPhdn.data is missing.');end
    d=r.data;req={'Xtr','Ytr','Xval','Yval','Xte','Yte'};
    for k=1:numel(req)
        if ~isfield(d,req{k})||isempty(d.(req{k}))
            error('resultPhdn.data.%s is missing or empty.',req{k});
        end
    end
end

function [X,Y]=get_ood_local(d)
    if isfield(d,'Xood')&&isfield(d,'Yood')&&~isempty(d.Xood)&&~isempty(d.Yood)
        X=d.Xood;Y=d.Yood;
    else
        X=[];Y=[];
    end
end

function [info,opts]=prepare_previous_reference_local(opts,d,XOod,YOod,currentN)
    info=empty_previous_info_local();
    if ~isfield(opts,'previousResult') || isempty(opts.previousResult)
        return;
    end
    prev=opts.previousResult;
    if ~isstruct(prev) || ~isfield(prev,'portableModel') || ...
            ~isstruct(prev.portableModel) || isempty(fieldnames(prev.portableModel))
        info.error='Previous EQL result has no portableModel.';
        return;
    end

    try
        info.available=true;
        info.result=prev;
        info.modelTrainingSamples=extract_training_sample_count_local(prev);
        info.statePath=extract_selected_state_path_local(prev);
        info.depth=getfield_default_local(prev,'depth',NaN);
        info.lambda=getfield_default_local(prev,'lambda',NaN);
        info.checkpoint=char(getfield_default_local(prev,'selectedCheckpoint',''));

        info.YTrainPred=predict_single_generator_dynamic_eql_baseline(prev,d.Xtr);
        info.YValPred=predict_single_generator_dynamic_eql_baseline(prev,d.Xval);
        info.YTestPred=predict_single_generator_dynamic_eql_baseline(prev,d.Xte);
        if isempty(XOod)
            info.YOodPred=zeros(0,size(d.Ytr,2));
        else
            info.YOodPred=predict_single_generator_dynamic_eql_baseline(prev,XOod);
        end

        info.trainMetrics=compute_regression_metrics(info.YTrainPred,d.Ytr);
        info.valMetrics=compute_regression_metrics(info.YValPred,d.Yval);
        info.testMetrics=compute_regression_metrics(info.YTestPred,d.Yte);
        if isempty(XOod)
            info.oodMetrics=empty_metrics_local(size(d.Ytr,2));
        else
            info.oodMetrics=compute_regression_metrics(info.YOodPred,YOod);
        end

        % Pass only scalar target metadata and a native selected state path to
        % the Python adapter. The previous result itself is not eligible for
        % unchanged current-N selection.
        opts.previousValidationMSE=info.valMetrics.mse;
        opts.previousSelectedStatePath=info.statePath;
        opts.previousSelectedDepth=info.depth;
        opts.previousSelectedLambda=info.lambda;
        opts.previousTrainingSampleCount=info.modelTrainingSamples;
        opts.currentTrainingSampleCount=currentN;
    catch ME
        info=empty_previous_info_local();
        info.error=sprintf('%s: %s',class(ME),ME.message);
    end
end

function result=attach_exact_n_protocol_local(fresh,prev,currentN)
    result=fresh;
    result.selectionRoute='exact_N_current_sample_adaptive_search';
    result.paperSampleEfficiencyProtocol= ...
        'exact_N_current_sample_training_with_adaptive_validation_search';
    result.previousModelRole= ...
        ['fixed-validation target and optional current-N warm-start only; ', ...
         'never unchanged substitution in the paper curve'];
    result.paperCurveModelSource='current_N_trained_model';
    result.paperCurveUsesExactN=true;
    result.modelTrainingSampleCount=currentN;
    result.exactNTrainingSampleCount=currentN;
    result.previousCandidateAvailable=prev.available;
    result.previousCandidateError=prev.error;
    result.previousCandidateValMSE=NaN;
    result.freshCandidateValMSE=result.valMetrics.mse;
    result.previousModelTrainingSampleCount=NaN;
    result.crossSampleSelectionMetric= ...
        'fixed_validation_mse_original_physical_output_scale';
    result.crossSampleCandidates=make_cross_sample_rows_local(prev,result,currentN);

    if ~prev.available
        result.monotoneEnvelopeSource='current_N_exact_model';
        result.monotoneEnvelopeModelTrainingSampleCount=currentN;
        result.monotoneEnvelopeValMSE=result.valMetrics.mse;
        result.monotoneEnvelopeIDRMSE=result.testMetrics.rmse;
        result.monotoneEnvelopeOODRMSE=getfield_default_local(result.oodMetrics,'rmse',NaN);
        return;
    end

    result.previousCandidateValMSE=prev.valMetrics.mse;
    result.previousModelTrainingSampleCount=prev.modelTrainingSamples;
    target=getfield_default_local(result,'strictCurrentNValidationTargetMSE',NaN);
    if ~isfinite(target)
        target=next_below_local(prev.valMetrics.mse);
        result.strictCurrentNValidationTargetMSE=target;
    end
    result.strictCurrentNImprovementAchieved= ...
        isfinite(result.valMetrics.mse) && result.valMetrics.mse<target;

    % Separate diagnostic only. It is never substituted into result.valMetrics,
    % result.testMetrics, rollout, or the paper sample-efficiency row.
    if prev.valMetrics.mse < result.valMetrics.mse
        result.monotoneEnvelopeSource='previous_smaller_N_reference';
        result.monotoneEnvelopeModelTrainingSampleCount=prev.modelTrainingSamples;
        result.monotoneEnvelopeValMSE=prev.valMetrics.mse;
        result.monotoneEnvelopeIDRMSE=prev.testMetrics.rmse;
        result.monotoneEnvelopeOODRMSE=getfield_default_local(prev.oodMetrics,'rmse',NaN);
    else
        result.monotoneEnvelopeSource='current_N_exact_model';
        result.monotoneEnvelopeModelTrainingSampleCount=currentN;
        result.monotoneEnvelopeValMSE=result.valMetrics.mse;
        result.monotoneEnvelopeIDRMSE=result.testMetrics.rmse;
        result.monotoneEnvelopeOODRMSE=getfield_default_local(result.oodMetrics,'rmse',NaN);
    end
    result.crossSampleCandidates=make_cross_sample_rows_local(prev,result,currentN);
end

function rows=make_cross_sample_rows_local(prev,current,currentN)
    rows=struct('source',{},'validationMSE',{},'paperCurveSelected',{}, ...
        'envelopeSelected',{},'modelTrainingSamples',{},'depth',{}, ...
        'lambda',{},'checkpoint',{},'activeUnits',{},'role',{});
    currentMSE=getfield_default_local(getfield_default_local(current,'valMetrics',struct()),'mse',NaN);
    previousWinsEnvelope=false;
    if prev.available
        previousWinsEnvelope=isfinite(prev.valMetrics.mse) && ...
            (~isfinite(currentMSE) || prev.valMetrics.mse<currentMSE);
        rows(end+1)=make_row_local('previous_smaller_N_reference',prev.result, ...
            prev.valMetrics.mse,false,previousWinsEnvelope, ...
            prev.modelTrainingSamples,'optimization target / diagnostic envelope'); %#ok<AGROW>
    end
    rows(end+1)=make_row_local('current_N_exact_model',current,currentMSE,true, ...
        ~previousWinsEnvelope,currentN,'paper curve: trained on all current-N samples'); %#ok<AGROW>
end

function row=make_row_local(source,r,mse,paperSelected,envelopeSelected,nModel,role)
    row=struct();
    row.source=source;
    row.validationMSE=mse;
    row.paperCurveSelected=logical(paperSelected);
    row.envelopeSelected=logical(envelopeSelected);
    row.modelTrainingSamples=nModel;
    row.depth=getfield_default_local(r,'depth',NaN);
    row.lambda=getfield_default_local(r,'lambda',NaN);
    row.checkpoint=char(getfield_default_local(r,'selectedCheckpoint',''));
    row.activeUnits=getfield_default_local(r,'nActiveUnits',NaN);
    row.role=role;
end

function path=extract_selected_state_path_local(r)
    path=char(getfield_default_local(r,'selectedStatePath',''));
    if ~isempty(path) && exist(path,'file')==2;return;end
    path='';
    if isfield(r,'pyResult') && isstruct(r.pyResult)
        path=char(getfield_default_local(r.pyResult,'selected_state_path',''));
        if ~isempty(path) && exist(path,'file')==2;return;end
        path='';
    end
    if isfield(r,'candidates') && ~isempty(r.candidates)
        candidates=normalize_candidates_local(r.candidates);
        for k=1:numel(candidates)
            c=candidates{k};
            if logical_scalar_local(getfield_default_local(c,'selected',false),false)
                candidatePath=char(getfield_default_local(c,'state_path',''));
                if ~isempty(candidatePath) && exist(candidatePath,'file')==2
                    path=candidatePath;return;
                end
            end
        end
    end
end

function n=extract_training_sample_count_local(r)
    n=get_nested_local(r,{'data','nTrain'},NaN);
    if ~isfinite(n);n=getfield_default_local(r,'modelTrainingSampleCount',NaN);end
    if ~isfinite(n);n=getfield_default_local(r,'exactNTrainingSampleCount',NaN);end
end

function items=normalize_candidates_local(raw)
    items={};
    if iscell(raw)
        for k=1:numel(raw)
            if isstruct(raw{k});items{end+1}=raw{k};end %#ok<AGROW>
        end
    elseif isstruct(raw)
        for k=1:numel(raw);items{end+1}=raw(k);end %#ok<AGROW>
    end
end

function info=empty_previous_info_local()
    info=struct('available',false,'error','','result',struct(), ...
        'modelTrainingSamples',NaN,'statePath','','depth',NaN,'lambda',NaN, ...
        'checkpoint','','YTrainPred',[],'YValPred',[],'YTestPred',[], ...
        'YOodPred',[],'trainMetrics',struct(),'valMetrics',struct(), ...
        'testMetrics',struct(),'oodMetrics',struct());
end

function m=empty_metrics_local(ny)
    m=struct('mse',NaN,'rmse',NaN,'mae',NaN,'maxAbsError',NaN, ...
        'mseByOutput',NaN(1,ny),'rmseByOutput',NaN(1,ny), ...
        'maeByOutput',NaN(1,ny),'maxAbsErrorByOutput',NaN(1,ny), ...
        'scaleByOutput',NaN(1,ny),'nrmseByOutput',NaN(1,ny), ...
        'nmaeByOutput',NaN(1,ny),'nmaxAbsErrorByOutput',NaN(1,ny), ...
        'nrmse',NaN,'nmae',NaN,'nmaxAbsError',NaN, ...
        'globalScale',NaN,'globalNRMSE',NaN,'globalNMAE',NaN);
end

function y=next_below_local(x)
    if ~isfinite(x);y=NaN;return;end
    y=x-max(eps(max(1,abs(x))),realmin('double'));
end

function value=get_nested_local(s,path,defaultValue)
    value=defaultValue;current=s;
    for k=1:numel(path)
        if ~isstruct(current)||~isfield(current,path{k})||isempty(current.(path{k}));return;end
        current=current.(path{k});
    end
    if (isnumeric(current)||islogical(current))&&isscalar(current);value=double(current);end
end

function tf=logical_scalar_local(v,d)
    if nargin<2;d=false;end
    if isempty(v);tf=d;elseif islogical(v)||isnumeric(v);tf=logical(v(1));else;tf=d;end
end

function v=getfield_default_local(s,n,d)
    if isstruct(s)&&isfield(s,n)&&~isempty(s.(n));v=s.(n);else;v=d;end
end

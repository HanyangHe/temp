function resultMlp = run_single_generator_dynamic_mlp_baseline_from_phdn_result(resultPhdn, mlpOpts)
%RUN_SINGLE_GENERATOR_DYNAMIC_MLP_BASELINE_FROM_PHDN_RESULT Case-local MLP sweep.
    if nargin < 2 || isempty(mlpOpts); mlpOpts = mlp_default_options(); end
    d = validate_shared_data_local(resultPhdn);
    [XOod,YOod] = get_ood_local(d);
    resultMlp = train_single_generator_dynamic_mlp_sweep_baseline( ...
        d.Xtr,d.Ytr,d.Xval,d.Yval,d.Xte,d.Yte,XOod,YOod,mlpOpts);
    resultMlp.dataSource='Exact PhDN/baseline data split';
    resultMlp.data=struct('nTrain',size(d.Xtr,1),'nVal',size(d.Xval,1), ...
        'nTest',size(d.Xte,1),'nOod',size(XOod,1));
    if isfield(d,'oodDomain'); resultMlp.data.oodDomain=d.oodDomain; end
    if getfield_default_local(mlpOpts,'verbose',true)
        print_single_generator_dynamic_mlp_result(resultMlp);
    end
end
function d=validate_shared_data_local(r)
    if ~isfield(r,'data')||isempty(r.data);error('resultPhdn.data is missing.');end
    d=r.data; req={'Xtr','Ytr','Xval','Yval','Xte','Yte'};
    for k=1:numel(req);if ~isfield(d,req{k})||isempty(d.(req{k}));error('resultPhdn.data.%s is missing or empty.',req{k});end;end
end
function [X,Y]=get_ood_local(d)
    if isfield(d,'Xood')&&isfield(d,'Yood')&&~isempty(d.Xood)&&~isempty(d.Yood);X=d.Xood;Y=d.Yood;else;X=[];Y=[];end
end
function v=getfield_default_local(s,n,d);if isstruct(s)&&isfield(s,n)&&~isempty(s.(n));v=s.(n);else;v=d;end;end

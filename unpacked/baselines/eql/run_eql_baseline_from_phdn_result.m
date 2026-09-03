function resultEql=run_eql_baseline_from_phdn_result(resultPhdn,eqlOpts)
%RUN_EQL_BASELINE_FROM_PHDN_RESULT Train official EQL on the shared raw split.
    if nargin<2||isempty(eqlOpts);eqlOpts=eql_default_options();end
    if ~isfield(resultPhdn,'data')||isempty(resultPhdn.data);error('resultPhdn.data is missing.');end
    d=resultPhdn.data;
    req={'Xtr','Ytr','Xval','Yval','Xte','Yte'};
    for k=1:numel(req)
        if ~isfield(d,req{k})||isempty(d.(req{k}));error('resultPhdn.data.%s is missing or empty.',req{k});end
    end
    if isfield(d,'Xood')&&isfield(d,'Yood')&&~isempty(d.Xood)&&~isempty(d.Yood)
        XOod=d.Xood;YOod=d.Yood;
    else
        XOod=[];YOod=[];
    end
    resultEql=train_eql_sweep_baseline( ...
        d.Xtr,d.Ytr,d.Xval,d.Yval,d.Xte,d.Yte,XOod,YOod,eqlOpts);
    resultEql.dataSource='Exact PhDN/baseline raw data split';
    resultEql.data=struct('nTrain',size(d.Xtr,1),'nVal',size(d.Xval,1), ...
        'nTest',size(d.Xte,1),'nOod',size(XOod,1));
    if isfield(d,'oodDomain');resultEql.data.oodDomain=d.oodDomain;end
    if getfield_default_local(eqlOpts,'verbose',true);print_eql_baseline_result(resultEql);end
end
function v=getfield_default_local(s,n,d)
    if isstruct(s)&&isfield(s,n)&&~isempty(s.(n));v=s.(n);else;v=d;end
end

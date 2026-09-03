function resultMlp = run_mlp_baseline_from_phdn_result(resultPhdn, mlpOpts)
%RUN_MLP_BASELINE_FROM_PHDN_RESULT Train MLP on the exact shared data split.
    if nargin < 2 || isempty(mlpOpts); mlpOpts = mlp_default_options(); end
    if ~isfield(resultPhdn,'data') || isempty(resultPhdn.data); error('resultPhdn.data is missing.'); end
    d=resultPhdn.data; req={'Xtr','Ytr','Xval','Yval','Xte','Yte'};
    for k=1:numel(req); if ~isfield(d,req{k}) || isempty(d.(req{k})); error('resultPhdn.data.%s is missing or empty.',req{k}); end; end
    if isfield(d,'Xood') && isfield(d,'Yood') && ~isempty(d.Xood) && ~isempty(d.Yood); XOod=d.Xood; YOod=d.Yood; else; XOod=[]; YOod=[]; end
    protocol=lower(strtrim(char(getfield_default_local(mlpOpts,'protocol','kan_feynman_sweep'))));
    switch protocol
        case 'kan_feynman_sweep'
            resultMlp=train_kan_paper_mlp_sweep_baseline(d.Xtr,d.Ytr,d.Xval,d.Yval,d.Xte,d.Yte,XOod,YOod,mlpOpts);
        case 'fixed_fitnet'
            resultMlp=train_mlp_regression_baseline(d.Xtr,d.Ytr,d.Xval,d.Yval,d.Xte,d.Yte,XOod,YOod,mlpOpts);
        otherwise
            error('Unknown MLP protocol: %s',protocol);
    end
    resultMlp.dataSource='Exact PhDN/baseline data split';
    resultMlp.data=struct('nTrain',size(d.Xtr,1),'nVal',size(d.Xval,1),'nTest',size(d.Xte,1),'nOod',size(XOod,1));
    if isfield(d,'oodDomain'); resultMlp.data.oodDomain=d.oodDomain; end
    if getfield_default_local(mlpOpts,'verbose',true); print_mlp_baseline_result(resultMlp); end
end
function v=getfield_default_local(s,n,d); if isstruct(s)&&isfield(s,n)&&~isempty(s.(n));v=s.(n);else;v=d;end;end

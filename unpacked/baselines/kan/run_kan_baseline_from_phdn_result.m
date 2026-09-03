function resultKan=run_kan_baseline_from_phdn_result(resultPhdn,kanOpts)
%RUN_KAN_BASELINE_FROM_PHDN_RESULT Train pruned KAN on exact shared data.
    if nargin<2||isempty(kanOpts);kanOpts=kan_default_options();end
    if ~isfield(resultPhdn,'data')||isempty(resultPhdn.data);error('resultPhdn.data is missing.');end
    d=resultPhdn.data;req={'Xtr','Ytr','Xval','Yval','Xte','Yte'};
    for k=1:numel(req);if ~isfield(d,req{k})||isempty(d.(req{k}));error('resultPhdn.data.%s is missing or empty.',req{k});end;end
    if isfield(d,'Xood')&&isfield(d,'Yood')&&~isempty(d.Xood)&&~isempty(d.Yood);XOod=d.Xood;YOod=d.Yood;else;XOod=[];YOod=[];end
    resultKan=train_pruned_kan_sweep_baseline(d.Xtr,d.Ytr,d.Xval,d.Yval,d.Xte,d.Yte,XOod,YOod,kanOpts);
    resultKan.dataSource='Exact PhDN/baseline data split';resultKan.data=struct('nTrain',size(d.Xtr,1),'nVal',size(d.Xval,1),'nTest',size(d.Xte,1),'nOod',size(XOod,1));
    if isfield(d,'oodDomain');resultKan.data.oodDomain=d.oodDomain;end
    if kanOpts.verbose;print_kan_baseline_result(resultKan);end
end

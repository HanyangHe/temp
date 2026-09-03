function [result,found,record] = load_feynman_dimless_method_result(resultsFile,iRound,caseName,methodField,methodLabel,compact)
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

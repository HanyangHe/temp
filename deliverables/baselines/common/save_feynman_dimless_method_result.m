function save_feynman_dimless_method_result(resultsFile,iRound,caseName,methodField,methodLabel,result,context)
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

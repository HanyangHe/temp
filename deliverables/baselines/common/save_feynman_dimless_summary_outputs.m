function save_feynman_dimless_summary_outputs(resultsFile,allResults,summaryRows,metadata)
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

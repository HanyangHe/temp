function [resultKan,allResults,summaryRows]=run_kan_comparison_after_phdn(task,resultPhdn,allResults,summaryRows,iCase,runFlag,kanOpts)
%RUN_KAN_COMPARISON_AFTER_PHDN One-line KAN integration helper.
    if nargin<6||isempty(runFlag);runFlag=true;end
    if nargin<7||isempty(kanOpts);kanOpts=make_default_kan_options_for_demo(1);end
    if nargin<3||isempty(allResults);allResults=struct();end
    if nargin<4;summaryRows=struct([]);end
    if nargin<5||isempty(iCase);iCase=1;end
    resultKan=[];
    if runFlag
        resultKan=run_kan_baseline_from_phdn_result(resultPhdn,kanOpts);allResults.(task.name).PhDN=resultPhdn;allResults.(task.name).KAN=resultKan;
    else
        allResults.(task.name).PhDN=resultPhdn;
    end
end

function resultKan=run_kan_baseline_from_task(task,opts,kanOpts)
%RUN_KAN_BASELINE_FROM_TASK Run KAN baseline without PhDN.
    if nargin<2||isempty(opts);opts=phdnn_default_options(task);end
    if nargin<3||isempty(kanOpts);kanOpts=make_default_kan_options_for_demo(1);end
    dataResult=make_baseline_data_result_from_task(task,opts);resultKan=run_kan_baseline_from_phdn_result(dataResult,kanOpts);
    resultKan.dataSource='Task-generated split (PhDN skipped)';
    resultKan.timeStats.dataGenerationTime=dataResult.timeStats.dataGenerationTime;
    resultKan.timeStats.splitTime=dataResult.timeStats.splitTime;resultKan.timeStats.oodDataGenerationTime=dataResult.timeStats.oodDataGenerationTime;
end

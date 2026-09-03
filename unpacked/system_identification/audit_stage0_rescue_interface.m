function audit_stage0_rescue_interface()
%AUDIT_STAGE0_RESCUE_INTERFACE Cheap static preflight for adaptive rescue plumbing.
% Run from anywhere after placing this file under <project>/system_identification.
% It performs no PySR training and catches the rescue-scope mismatch that could
% otherwise appear only after the expensive base Stage-0 restarts finish.

    thisFile = mfilename('fullpath');
    projectRoot = fileparts(fileparts(thisFile));

    coreFile = fullfile(projectRoot,'core','stage0', ...
        'run_phdn_per_output_pysr_stage0.m');
    trainFile = fullfile(projectRoot,'baselines','sr', ...
        'train_official_pysr_baseline.m');
    adapterFile = fullfile(projectRoot,'baselines','sr', ...
        'pysr_official_adapter.py');

    required = {coreFile,trainFile,adapterFile};
    for k = 1:numel(required)
        assert(exist(required{k},'file')==2, ...
            'Missing required rescue-interface file: %s',required{k});
    end

    coreText = fileread(coreFile);
    trainText = fileread(trainFile);
    adapterText = fileread(adapterFile);

    badCoreAssignment = regexp(coreText, ...
        'rescueOpts\.initialGuessScope\s*=\s*''single_output_rescue''', ...
        'once');
    assert(isempty(badCoreAssignment), ...
        ['STALE rescue scope detected in run_phdn_per_output_pysr_stage0.m. ', ...
         'Install the general rescue-scope hotfix before an expensive run.']);

    assert(contains(coreText, ...
        'rescueOpts.initialGuessScope = ''shared_all_unresolved_outputs'';'), ...
        'Canonical single-output rescue scope assignment is missing.');

    assert(contains(trainText,'''single_output_rescue''') && ...
        contains(trainText,'''shared_all_unresolved_outputs'''), ...
        'MATLAB PySR wrapper does not contain the legacy rescue-scope guard.');

    assert(contains(adapterText,'legacy_single_output_rescue_scope_normalized') && ...
        contains(adapterText,'valid only for a true single-output targeted-rescue fit'), ...
        'Python PySR adapter does not contain the defensive rescue-scope guard.');

    fprintf('\nStage-0 adaptive-rescue interface audit: PASS\n');
    fprintf('  core controller : canonical scope used for rescue\n');
    fprintf('  MATLAB wrapper  : stale rescue alias normalized\n');
    fprintf('  Python adapter  : stale alias accepted only for nOutputs=1\n');
    fprintf(['This global path is shared by Generator main/noise and ', ...
        'Lorenz main/noise Stage-0 rescue runs.\n\n']);
end

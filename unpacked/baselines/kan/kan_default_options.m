function kanOpts = kan_default_options()
%KAN_DEFAULT_OPTIONS Defaults for the official-pyKAN Feynman pruned baseline.
%
% Training order follows bundled pyKAN kan/experiment.py::runner1 while the
% sweep ranges follow Appendix P:
%   1) fixed width 5; sweep KAN depth 2:6 and lambda={1e-2,1e-3};
%   2) initialize at G=3 and run one sparse LBFGS fit;
%   3) immediately call the official model.prune() once;
%   4) refine the pruned model through G=[5,10,20,50,100,200], with one
%      ordinary lamb=0 official model.fit() call at each attempted grid;
%   5) evaluate the framework validation split after every grid; at the first
%      validation-RMSE increase, stop and restore the previous best grid;
%   6) no warm-up fit, repeated sparsification, recovery fit, or custom KAN
%      implementation is added.
%
% Only the supplied sample matrices and the train/validation/test/OOD split
% differ from pyKAN's generated Feynman data. For the broadened heterogeneous
% domains, input and output z-score statistics are fitted on the training split
% only. KAN trains on standardized X/Y, while all predictions and reported
% metrics are inverse-transformed to the original output units.
    kanDir=fileparts(mfilename('fullpath')); projectRoot=fileparts(fileparts(kanDir));
    kanOpts=struct();
    kanOpts.protocol='official_pykan_feynman_pruned_refinement_validation_early_stop';
    kanOpts.seed=1;
    kanOpts.width=5;
    kanOpts.depthList=2:6;
    kanOpts.minimumDepth=min(kanOpts.depthList); % updated across nested N sweep
    kanOpts.depthConvention='number_of_KAN_layers';
    kanOpts.gridList=[3,5,10,20,50,100,200];
    kanOpts.splineOrder=3;
    kanOpts.sparsificationLambdaList=[1e-2,1e-3];
    kanOpts.stepsPerGrid=200;
    kanOpts.optimizer='LBFGS';
    kanOpts.learningRate=1.0;
    kanOpts.pruneNodeThreshold=1e-2;
    kanOpts.pruneEdgeThreshold=3e-2;
    kanOpts.selectionMetric='validation_mse';
    kanOpts.gridEarlyStop=true;
    kanOpts.gridEarlyStopPatience=1; % stop at the first meaningful validation rise
    kanOpts.gridEarlyStopRelativeTolerance=0.0; % 0 = any strict rise; MSE/RMSE ordering is identical
    kanOpts.dtype='float32';
    kanOpts.device='cpu';
    kanOpts.torchNumThreads=0; % 0 = leave the official PyTorch setting unchanged
    kanOpts.normalizeInputs=true;  % train-split z-score; reused for val/test/OOD
    kanOpts.normalizeOutputs=true; % train on standardized Y; predictions are denormalized before metrics
    kanOpts.pythonExe='python';
    kanOpts.pykanRoot=fullfile(projectRoot,'third_party','pykan');
    kanOpts.workRoot=fullfile(projectRoot,'tmp','kan_official_feynman_runs');
    kanOpts.displaySweepTable=true;
    kanOpts.verbose=true;
end

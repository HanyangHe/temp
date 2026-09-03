function [resultPhdn,resultStage0SR] = ...
    attach_soft_saturated_lorenz96_stage0_sr_ablation(resultPhdn)
%ATTACH_SOFT_SATURATED_LORENZ96_STAGE0_SR_ABLATION Restore the SR ablation.
%
% Some branches already attach result.ablations.stage0SR; others retain all
% Stage-0 information but omit the convenience field.  Reconstruct it without
% rerunning PySR when needed.

    if isfield(resultPhdn,'ablations') && ...
            isstruct(resultPhdn.ablations) && ...
            isfield(resultPhdn.ablations,'stage0SR')
        resultStage0SR = resultPhdn.ablations.stage0SR;
        return;
    end

    resultStage0SR = make_stage0_sr_ablation_result(resultPhdn);
    if ~isfield(resultPhdn,'ablations') || ...
            ~isstruct(resultPhdn.ablations)
        resultPhdn.ablations = struct();
    end
    resultPhdn.ablations.stage0SR = resultStage0SR;
end

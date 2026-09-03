function [isComplete,report] = validate_soft_saturated_lorenz96_noise_rows(rows,noiseLevels,numRounds,requiredMethods)
%VALIDATE_SOFT_SATURATED_LORENZ96_NOISE_ROWS Exact (method,round,rho) coverage check.

    report = struct('missing',{{}},'duplicates',{{}});
    isComplete = true;
    tol = 1e-14;
    if isempty(rows) || ~isstruct(rows) || ~isfield(rows,'method') || ...
            ~isfield(rows,'roundIndex') || ~isfield(rows,'noiseLevel')
        isComplete = false;
        report.missing = {'row structure is empty or lacks required fields'};
        return;
    end

    for iMethod = 1:numel(requiredMethods)
        method = char(requiredMethods{iMethod});
        for iRound = 1:numRounds
            for iNoise = 1:numel(noiseLevels)
                rho = double(noiseLevels(iNoise));
                mask = strcmpi({rows.method},method) & ...
                    [rows.roundIndex] == iRound & ...
                    abs([rows.noiseLevel]-rho) <= tol;
                n = sum(mask);
                if n == 0
                    isComplete = false;
                    report.missing{end+1} = sprintf('%s | round=%d | rho=%.6g (%.3g%%)', ...
                        method,iRound,rho,100*rho); %#ok<AGROW>
                elseif n > 1
                    isComplete = false;
                    report.duplicates{end+1} = sprintf('%s | round=%d | rho=%.6g has %d rows', ...
                        method,iRound,rho,n); %#ok<AGROW>
                end
            end
        end
    end
end

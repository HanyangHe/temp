function selectedRows = filter_soft_saturated_lorenz96_noise_rows(rows,noiseLevels,numRounds,methods)
%FILTER_SOFT_SATURATED_LORENZ96_NOISE_ROWS Select one protocol from mixed stored rho rows.

    if nargin < 4; methods = {}; end
    if isempty(rows); selectedRows = rows; return; end
    if ~isstruct(rows) || ~isfield(rows,'noiseLevel') || ~isfield(rows,'roundIndex')
        error('Noise rows must contain noiseLevel and roundIndex fields.');
    end

    noiseLevels = reshape(double(noiseLevels),1,[]);
    tol = 1e-14;
    keep = false(1,numel(rows));
    for i = 1:numel(rows)
        rho = double(rows(i).noiseLevel);
        inNoiseGrid = any(abs(noiseLevels-rho) <= tol);
        inRoundGrid = rows(i).roundIndex >= 1 && rows(i).roundIndex <= numRounds;
        inMethodSet = true;
        if ~isempty(methods) && isfield(rows,'method')
            inMethodSet = any(strcmpi(char(rows(i).method),methods));
        end
        keep(i) = inNoiseGrid && inRoundGrid && inMethodSet;
    end
    selectedRows = rows(keep);
end

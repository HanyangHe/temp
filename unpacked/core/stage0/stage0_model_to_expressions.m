function expressions = stage0_model_to_expressions(termNames, Xi, zeroTolerance, activeMask)
%STAGE0_MODEL_TO_EXPRESSIONS Convert a fitted fixed-SINDy model to output equations.

    if nargin < 3; zeroTolerance = 1e-12; end
    if nargin < 4 || isempty(activeMask); activeMask = abs(Xi) > zeroTolerance; end
    expressions = cell(1,size(Xi,2));
    for k = 1:size(Xi,2)
        parts = {};
        for j = 1:size(Xi,1)
            c = Xi(j,k);
            if ~activeMask(j,k) || ~isfinite(c)
                continue;
            end
            term = strrep(strtrim(char(termNames{j})), ' ', '');
            if strcmp(term,'1')
                piece = sprintf('(%.16g)', c);
            elseif abs(c - 1) <= zeroTolerance
                piece = sprintf('(%s)', term);
            elseif abs(c + 1) <= zeroTolerance
                piece = sprintf('-(%s)', term);
            else
                piece = sprintf('(%.16g)*(%s)', c, term);
            end
            parts{end+1} = piece; %#ok<AGROW>
        end
        if isempty(parts)
            expressions{k} = '0';
        else
            expressions{k} = strjoin(parts, '+');
            expressions{k} = strrep(expressions{k}, '+-', '-');
        end
    end
end

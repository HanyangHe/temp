function Y = predict_single_generator_dynamic_stage0_sr(methodResult, X)
%PREDICT_SINGLE_GENERATOR_DYNAMIC_STAGE0_SR Evaluate selected Stage-0 expressions.
%
% This case-local helper makes the mixed Stage-0 model (per-output SINDy
% bypass plus official-PySR expressions) callable at arbitrary states for
% ODE rollout.  It evaluates the exact expressions already selected during
% the PhDN Stage-0 run; it does not launch another SR fit and does not use
% the Stage-1/2 refined PhDN coefficients.
%
% X is N-by-4 with x1=delta, x2=domega, x3=Eqp, x4=Efd.

    validateattributes(X, {'numeric'}, {'2d','ncols',4,'real','finite'}, ...
        mfilename, 'X');

    expressions = get_selected_expressions_local(methodResult);
    if numel(expressions) ~= 4
        error(['SingleGeneratorDynamic Stage0-SR rollout requires four ', ...
            'selected output expressions; received %d.'], numel(expressions));
    end

    x1 = X(:,1); %#ok<NASGU>
    x2 = X(:,2); %#ok<NASGU>
    x3 = X(:,3); %#ok<NASGU>
    x4 = X(:,4); %#ok<NASGU>
    n = size(X,1);
    Y = zeros(n,4);

    for j = 1:4
        expression = prepare_expression_local(expressions{j});
        validate_expression_local(expression);
        try
            value = eval(expression); %#ok<EVLDIR>
        catch ME
            error('Failed to evaluate Stage0-SR output y%d: %s\nExpression: %s', ...
                j, ME.message, expression);
        end

        if isscalar(value)
            value = repmat(value,n,1);
        else
            value = reshape(value,[],1);
        end
        if numel(value) ~= n || any(~isfinite(value))
            error(['Stage0-SR output y%d returned an invalid prediction ', ...
                '(expected %d finite values, received %d).'], ...
                j,n,numel(value));
        end
        Y(:,j) = value;
    end
end

function expressions = get_selected_expressions_local(methodResult)
    expressions = {};
    candidates = {'selectedExpressions','bestExpressions','stage0Expressions'};
    for i = 1:numel(candidates)
        fieldName = candidates{i};
        if isstruct(methodResult) && isfield(methodResult,fieldName) && ...
                ~isempty(methodResult.(fieldName))
            expressions = methodResult.(fieldName);
            break;
        end
    end
    if isempty(expressions) && isstruct(methodResult) && ...
            isfield(methodResult,'stage0Info') && isstruct(methodResult.stage0Info)
        for i = 1:numel(candidates)
            fieldName = candidates{i};
            if isfield(methodResult.stage0Info,fieldName) && ...
                    ~isempty(methodResult.stage0Info.(fieldName))
                expressions = methodResult.stage0Info.(fieldName);
                break;
            end
        end
    end
    if ischar(expressions) || isstring(expressions)
        expressions = cellstr(expressions);
    end
    if ~iscell(expressions)
        error('Stage0-SR selected expressions are unavailable or malformed.');
    end
    expressions = reshape(expressions,1,[]);
end

function expression = prepare_expression_local(expression)
    expression = char(string(expression));
    expression = strtrim(expression);

    % Official PySR may emit Python power syntax and protected helper names.
    expression = strrep(expression,'**','^');
    expression = regexprep(expression,'\<sqrt_abs\s*\(', ...
        'sr_sqrt_abs_local(');
    expression = regexprep(expression,'\<sqrtabs\s*\(', ...
        'sr_sqrt_abs_local(');
    expression = regexprep(expression,'\<square\s*\(', ...
        'sr_square_local(');
    expression = regexprep(expression,'\<cube\s*\(', ...
        'sr_cube_local(');
    expression = regexprep(expression,'\<inv\s*\(', ...
        'sr_inv_local(');
    expression = regexprep(expression,'\<Abs\s*\(', 'abs(');

    % Vectorize binary arithmetic while leaving already-vectorized operators
    % unchanged.  The expressions are evaluated on N-by-1 state columns.
    expression = regexprep(expression,'(?<!\.)\^','.^');
    expression = regexprep(expression,'(?<!\.)\*','.*');
    expression = regexprep(expression,'(?<!\.)/','./');
end

function validate_expression_local(expression)
    % Expressions originate from the internal SINDy/PySR result, but reject
    % statement separators and workspace/file-system constructs before eval.
    forbidden = {';','@','[',']','{','}','''','"','=',':','!','~','\\'};
    for i = 1:numel(forbidden)
        if contains(expression,forbidden{i})
            error('Unsupported token "%s" in Stage0-SR expression.', ...
                forbidden{i});
        end
    end

    % Remove allowed identifiers and verify no unknown alphabetic token is
    % left.  This keeps the evaluator restricted to the task variables,
    % elementary operators, and case-local protected helpers.
    allowed = {'x1','x2','x3','x4','sin','cos','tan','exp','log','sqrt','abs', ...
        'sr_square_local','sr_cube_local','sr_inv_local','sr_sqrt_abs_local', ...
        'pi','Inf','NaN','e'};
    residue = expression;
    for i = 1:numel(allowed)
        residue = regexprep(residue, ['\<' allowed{i} '\>'], '');
    end
    residue = regexprep(residue, '[0-9eE\.\+\-\*\/\^\(\),\s]', '');
    if ~isempty(residue)
        error('Unknown token(s) in Stage0-SR expression: %s', residue);
    end
end

function y = sr_square_local(x)
    y = x.^2;
end

function y = sr_cube_local(x)
    y = x.^3;
end

function y = sr_inv_local(x)
    y = 1./x;
end

function y = sr_sqrt_abs_local(x)
    y = sqrt(abs(x));
end

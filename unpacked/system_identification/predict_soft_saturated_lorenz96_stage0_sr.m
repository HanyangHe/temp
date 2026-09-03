function Y = predict_soft_saturated_lorenz96_stage0_sr(methodResult, X)
%PREDICT_SOFT_SATURATED_LORENZ96_STAGE0_SR Evaluate selected Stage-0 cores.

    nState = infer_state_count_local(methodResult,X);
    validateattributes(X, {'numeric'}, {'2d','ncols',nState,'real','finite'}, ...
        mfilename, 'X');

    expressions = get_selected_expressions_local(methodResult);
    if numel(expressions) ~= nState
        error(['SoftSaturatedLorenz96 Stage0-SR rollout requires %d selected ', ...
            'output expressions; received %d.'],nState,numel(expressions));
    end

    for k = 1:nState
        eval(sprintf('x%d = X(:,%d);',k,k)); %#ok<EVLDIR>
    end
    n = size(X,1);
    Y = zeros(n,nState);

    for j = 1:nState
        expression = prepare_expression_local(expressions{j});
        validate_expression_local(expression,nState);
        try
            value = eval(expression); %#ok<EVLDIR>
        catch ME
            error('Failed to evaluate Stage0-SR output y%d: %s\nExpression: %s', ...
                j,ME.message,expression);
        end
        if isscalar(value); value = repmat(value,n,1); else; value = reshape(value,[],1); end
        if numel(value) ~= n || any(~isfinite(value))
            error('Stage0-SR output y%d returned an invalid prediction.',j);
        end
        Y(:,j) = value;
    end
end

function nState = infer_state_count_local(r,X)
    nState = size(X,2);
    if isstruct(r) && isfield(r,'task') && isstruct(r.task) && ...
            isfield(r.task,'nx') && isfinite(r.task.nx)
        nState = double(r.task.nx);
    elseif isstruct(r) && isfield(r,'data') && isfield(r.data,'Xtr') && ...
            ~isempty(r.data.Xtr)
        nState = size(r.data.Xtr,2);
    end
end

function expressions = get_selected_expressions_local(methodResult)
    expressions = {};
    candidates = {'selectedExpressions','bestExpressions','stage0Expressions'};
    for i = 1:numel(candidates)
        fieldName = candidates{i};
        if isstruct(methodResult) && isfield(methodResult,fieldName) && ...
                ~isempty(methodResult.(fieldName))
            expressions = methodResult.(fieldName); break;
        end
    end
    if isempty(expressions) && isstruct(methodResult) && ...
            isfield(methodResult,'stage0Info') && isstruct(methodResult.stage0Info)
        for i = 1:numel(candidates)
            fieldName = candidates{i};
            if isfield(methodResult.stage0Info,fieldName) && ...
                    ~isempty(methodResult.stage0Info.(fieldName))
                expressions = methodResult.stage0Info.(fieldName); break;
            end
        end
    end
    if ischar(expressions) || isstring(expressions); expressions = cellstr(expressions); end
    if ~iscell(expressions); error('Stage0-SR expressions are unavailable or malformed.'); end
    expressions = reshape(expressions,1,[]);
end

function expression = prepare_expression_local(expression)
    expression = strtrim(char(string(expression)));
    expression = strrep(expression,'**','^');
    expression = regexprep(expression,'\<sqrt_abs\s*\(','sr_sqrt_abs_local(');
    expression = regexprep(expression,'\<sqrtabs\s*\(','sr_sqrt_abs_local(');
    expression = regexprep(expression,'\<square\s*\(','sr_square_local(');
    expression = regexprep(expression,'\<cube\s*\(','sr_cube_local(');
    expression = regexprep(expression,'\<inv\s*\(','sr_inv_local(');
    expression = regexprep(expression,'\<Abs\s*\(','abs(');
    expression = regexprep(expression,'(?<!\.)\^','.^');
    expression = regexprep(expression,'(?<!\.)\*','.*');
    expression = regexprep(expression,'(?<!\.)/','./');
end

function validate_expression_local(expression,nState)
    forbidden = {';','@','[',']','{','}','''','"','=',':','!','~','\\'};
    for i = 1:numel(forbidden)
        if contains(expression,forbidden{i})
            error('Unsupported token "%s" in Stage0-SR expression.',forbidden{i});
        end
    end
    allowed = arrayfun(@(k) sprintf('x%d',k),1:nState,'UniformOutput',false);
    allowed = [allowed,{'sin','cos','tanh','tan','exp','log','sqrt','abs', ...
        'sr_square_local','sr_cube_local','sr_inv_local','sr_sqrt_abs_local', ...
        'pi','Inf','NaN','e'}];
    residue = expression;
    [~,order] = sort(cellfun(@numel,allowed),'descend');
    for i = order
        residue = regexprep(residue,['\<' allowed{i} '\>'],'');
    end
    residue = regexprep(residue,'[0-9eE\.\+\-\*\/\^\(\),\s]','');
    if ~isempty(residue); error('Unknown token(s) in Stage0-SR expression: %s',residue); end
end

function y = sr_square_local(x); y = x.^2; end
function y = sr_cube_local(x); y = x.^3; end
function y = sr_inv_local(x); y = 1./x; end
function y = sr_sqrt_abs_local(x); y = sqrt(abs(x)); end

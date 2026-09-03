function Phi = evaluate_compiled_value_terms(plan, H)
%EVALUATE_COMPILED_VALUE_TERMS Evaluate a pre-parsed value-only term plan.
%
% H is inputDim-by-N. No dictionary text parsing and no Jacobian evaluation
% occurs for compiled terms. Unsupported terms use the original evaluator as
% a correctness fallback.

    if ~isnumeric(H) || ndims(H) ~= 2
        error('evaluate_compiled_value_terms:InvalidInput', ...
            'H must be a two-dimensional numeric matrix.');
    end

    nTerms = numel(plan.names);
    nSamples = size(H, 2);
    Phi = zeros(nTerms, nSamples);

    safety = struct('eps', plan.eps);
    if isfield(plan, 'arch') && isstruct(plan.arch) && ...
            isfield(plan.arch, 'safety') && isstruct(plan.arch.safety)
        safety = plan.arch.safety;
        if ~isfield(safety, 'eps') || isempty(safety.eps)
            safety.eps = plan.eps;
        end
    end

    for k = 1:nTerms
        if plan.fallback(k)
            [row, ~, ~, ~] = evaluate_explicit_case_terms(H, plan.names(k), ...
                safety, plan.arch, plan.rawMask(k));
            Phi(k, :) = row;
        else
            Phi(k, :) = evaluate_node_local(plan.nodes{k}, H, ...
                plan.rawMask(k), plan.eps);
        end
    end

    Phi(~isfinite(Phi)) = 0;
end

function value = evaluate_node_local(node, H, useRawSemantics, epsVal)
    nSamples = size(H, 2);

    switch node.kind
        case 'constant'
            value = node.value .* ones(1, nSamples);

        case 'variable'
            if node.index < 1 || node.index > size(H, 1)
                error('Variable index v%d exceeds input dimension %d.', ...
                    node.index, size(H, 1));
            end
            value = H(node.index, :);

        case 'sum'
            value = zeros(1, nSamples);
            for k = 1:numel(node.children)
                child = evaluate_node_local(node.children{k}, H, ...
                    useRawSemantics, epsVal);
                value = value + node.signs(k) .* child;
            end

        case 'neg'
            value = -evaluate_node_local(node.children{1}, H, ...
                useRawSemantics, epsVal);

        case 'muldiv'
            value = evaluate_node_local(node.children{1}, H, ...
                useRawSemantics, epsVal);
            for k = 2:numel(node.children)
                right = evaluate_node_local(node.children{k}, H, ...
                    useRawSemantics, epsVal);
                if node.ops(k-1) == '*'
                    value = value .* right;
                else
                    if useRawSemantics
                        denominator = right;
                    else
                        denominator = protected_denominator_local(right, epsVal);
                    end
                    value = value ./ denominator;
                end
            end

        case 'power'
            base = evaluate_node_local(node.children{1}, H, ...
                useRawSemantics, epsVal);
            value = base .^ node.exponent;

        case 'binary'
            left = evaluate_node_local(node.children{1}, H, ...
                useRawSemantics, epsVal);
            right = evaluate_node_local(node.children{2}, H, ...
                useRawSemantics, epsVal);
            error('Unsupported compiled binary operator: %s', node.name);

        case 'unary'
            argument = evaluate_node_local(node.children{1}, H, ...
                useRawSemantics, epsVal);
            value = evaluate_unary_local(node.name, argument, ...
                useRawSemantics, epsVal);

        otherwise
            error('Unsupported compiled AST node: %s', node.kind);
    end
end

function value = evaluate_unary_local(name, argument, useRawSemantics, epsVal)
    switch lower(name)
        case {'re', 'real', 'conj', 'conjugate'}
            value = argument;

        case {'im', 'imag'}
            value = zeros(size(argument));

        case 'inv'
            if useRawSemantics
                denominator = argument;
            else
                denominator = protected_denominator_local(argument, epsVal);
            end
            value = 1 ./ denominator;

        case {'square', 'sqr'}
            value = argument.^2;

        case 'cube'
            value = argument.^3;

        case 'abs'
            value = abs(argument);

        case 'sqrt_abs'
            if useRawSemantics
                value = sqrt(abs(argument));
            else
                value = sqrt(max(abs(argument), epsVal));
            end

        case 'sqrt'
            if useRawSemantics
                value = nan(size(argument));
                valid = argument >= 0;
                value(valid) = sqrt(argument(valid));
            else
                value = sqrt(max(argument, epsVal));
            end

        case 'exp'
            if useRawSemantics
                value = exp(argument);
            else
                value = exp(min(max(argument, -50), 50));
            end

        case 'sin'
            value = sin(argument);

        case 'cos'
            value = cos(argument);

        case 'tanh'
            value = tanh(argument);

        case 'asin'
            if useRawSemantics
                value = nan(size(argument));
                valid = abs(argument) <= 1;
                value(valid) = asin(argument(valid));
            else
                clipped = min(max(argument, -1 + epsVal), 1 - epsVal);
                value = asin(clipped);
            end

        case 'log'
            if useRawSemantics
                value = nan(size(argument));
                valid = argument > 0;
                value(valid) = log(argument(valid));
            else
                protected = protected_denominator_local(argument, epsVal);
                value = log(abs(protected));
            end

        otherwise
            chebOrder = parse_chebyshev_order_local(name);
            if isempty(chebOrder)
                error('Unsupported compiled unary operator: %s', name);
            end
            value = chebyshev_value_local(argument, chebOrder);
    end
end

function denominator = protected_denominator_local(value, epsVal)
    signValue = sign(value);
    signValue(signValue == 0) = 1;
    denominator = signValue .* max(abs(value), epsVal);
end

function order = parse_chebyshev_order_local(name)
    token = regexp(char(name), '^[Tt](\d+)$', 'tokens', 'once');
    if isempty(token)
        order = [];
        return;
    end
    order = str2double(token{1});
    if ~isfinite(order) || order < 0 || abs(order - round(order)) > 0
        order = [];
    else
        order = round(order);
    end
end

function value = chebyshev_value_local(x, order)
    if order == 0
        value = ones(size(x));
        return;
    elseif order == 1
        value = x;
        return;
    end

    previous = ones(size(x));
    current = x;
    for k = 1:(order - 1)
        next = 2 .* x .* current - previous;
        previous = current;
        current = next;
    end
    value = current;
end

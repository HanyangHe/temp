function report = prepare_stage0_guess_linear_coefficients(task, samplingPlan, outputIndex, basisExpressions, userOptions)
%PREPARE_STAGE0_GUESS_LINEAR_COEFFICIENTS Fit outer linear coefficients for an SR guess.
%
% report = prepare_stage0_guess_linear_coefficients(task, samplingPlan, ...
%     outputIndex, basisExpressions)
%
% The function uses the case task and the task.sampleFcn configured by the
% corresponding demo. Therefore, it reproduces that demo's training sample
% stream rather than creating an unrelated auxiliary data set.
%
% Inputs
%   task             Case task after the demo has assigned task.sampleFcn.
%   samplingPlan     Demo sampling plan with nTrain, nValidation, and nTest.
%   outputIndex      Target derivative/output index, e.g. 4 for y4.
%   basisExpressions Cell array of coefficient-free basis expressions, e.g.
%                       {'1','x4','sqrt(square(x3)+square(x1))'}
%   userOptions      Optional struct:
%       .rngSeed                  RNG state used before demo sampling (default 1)
%       .ridgeLambda              nonnegative ridge coefficient (default 0)
%       .coefficientZeroTolerance printed-expression threshold (default 1e-12)
%       .coefficientFormat        sprintf format (default '%.16g')
%       .printReport              logical (default true)
%
% Output report fields include Xtrain, ytrain, Phi, coefficients, trainMSE,
% trainRMSE, conditionNumber, rank, basisExpressions, and guessExpression.
%
% This is a preparation/diagnostic tool only. It does not modify the demo's
% Stage0SRInitialGuesses automatically; the printed expression is intended
% for review and manual insertion by the user.

    if nargin < 5 || isempty(userOptions); userOptions = struct(); end
    validateattributes(task, {'struct'}, {'scalar'}, mfilename, 'task');
    validateattributes(samplingPlan, {'struct'}, {'scalar'}, mfilename, 'samplingPlan');
    validateattributes(outputIndex, {'numeric'}, {'scalar','integer','positive'}, ...
        mfilename, 'outputIndex');
    basisExpressions = normalize_basis_list_local(basisExpressions);
    if isempty(basisExpressions)
        error('basisExpressions must contain at least one expression.');
    end
    requiredPlan = {'nTrain','nValidation','nTest'};
    for k = 1:numel(requiredPlan)
        if ~isfield(samplingPlan,requiredPlan{k}) || isempty(samplingPlan.(requiredPlan{k}))
            error('samplingPlan.%s is required.',requiredPlan{k});
        end
    end
    if ~isfield(task,'sampleFcn') || isempty(task.sampleFcn)
        error(['task.sampleFcn is missing. Configure the task exactly as in the ', ...
            'corresponding case demo before calling this tool.']);
    end
    if ~isfield(task,'rhsFcn') || isempty(task.rhsFcn)
        error('task.rhsFcn is required to generate derivative targets.');
    end
    if ~isfield(task,'domain') || isempty(task.domain)
        error('task.domain is required for demo-matched sampling.');
    end
    if isfield(task,'nx') && outputIndex > task.nx
        error('outputIndex=%d exceeds task.nx=%d.',outputIndex,task.nx);
    end

    rngSeed = get_option_local(userOptions,'rngSeed',1);
    ridgeLambda = get_option_local(userOptions,'ridgeLambda',0);
    zeroTol = get_option_local(userOptions,'coefficientZeroTolerance',1e-12);
    coeffFormat = char(get_option_local(userOptions,'coefficientFormat','%.16g'));
    printReport = logical(get_option_local(userOptions,'printReport',true));
    validateattributes(rngSeed, {'numeric'}, {'scalar','integer','nonnegative'});
    validateattributes(ridgeLambda, {'numeric'}, {'scalar','real','finite','nonnegative'});
    validateattributes(zeroTol, {'numeric'}, {'scalar','real','finite','nonnegative'});

    nTrain = samplingPlan.nTrain;
    nTotal = samplingPlan.nTrain + samplingPlan.nValidation + samplingPlan.nTest;

    oldRng = rng;
    rngCleanup = onCleanup(@() rng(oldRng)); %#ok<NASGU>
    rng(rngSeed,'twister');
    Xcombined = task.sampleFcn(nTotal,task.domain);
    splitIndex = randperm(nTotal);
    Xtrain = Xcombined(splitIndex(1:nTrain),:);
    Ytrain = task.rhsFcn(Xtrain);
    if size(Ytrain,1) ~= nTrain || outputIndex > size(Ytrain,2)
        error('task.rhsFcn returned an unexpected target matrix size.');
    end
    ytrain = Ytrain(:,outputIndex);

    Phi = evaluate_basis_matrix_local(basisExpressions,Xtrain);
    validRows = all(isfinite(Phi),2) & isfinite(ytrain);
    if ~all(validRows)
        warning('%d/%d rows were removed because a basis or target was nonfinite.', ...
            nnz(~validRows),numel(validRows));
        PhiFit = Phi(validRows,:);
        yFit = ytrain(validRows);
    else
        PhiFit = Phi;
        yFit = ytrain;
    end
    if size(PhiFit,1) < size(PhiFit,2)
        warning('The fit is underdetermined: %d rows for %d basis columns.', ...
            size(PhiFit,1),size(PhiFit,2));
    end

    gram = PhiFit.'*PhiFit;
    rhs = PhiFit.'*yFit;
    if ridgeLambda > 0
        scale = max(1,trace(gram)/max(1,size(gram,1)));
        coefficients = (gram + ridgeLambda*scale*eye(size(gram)))\rhs;
    else
        coefficients = PhiFit\yFit;
    end
    prediction = PhiFit*coefficients;
    residual = yFit-prediction;
    trainMSE = mean(residual.^2);
    trainRMSE = sqrt(trainMSE);
    matrixRank = rank(PhiFit);
    singularValues = svd(PhiFit,'econ');
    if isempty(singularValues) || singularValues(end) <= eps(max(singularValues))*max(size(PhiFit))
        conditionNumber = Inf;
    else
        conditionNumber = singularValues(1)/singularValues(end);
    end
    guessExpression = build_guess_expression_local( ...
        coefficients,basisExpressions,zeroTol,coeffFormat);

    report = struct();
    report.taskName = getfield_default_local(task,'name','unnamed_task');
    report.outputIndex = outputIndex;
    report.samplingPlan = samplingPlan;
    report.rngSeed = rngSeed;
    report.ridgeLambda = ridgeLambda;
    report.basisExpressions = basisExpressions;
    report.coefficients = coefficients;
    report.guessExpression = guessExpression;
    report.trainMSE = trainMSE;
    report.trainRMSE = trainRMSE;
    report.conditionNumber = conditionNumber;
    report.rank = matrixRank;
    report.nBasis = size(PhiFit,2);
    report.nTrainingRows = size(PhiFit,1);
    report.validRowMask = validRows;
    report.Xtrain = Xtrain;
    report.ytrain = ytrain;
    report.Phi = Phi;

    if printReport
        fprintf('\n============================================================\n');
        fprintf('Stage-0 guess outer-coefficient preparation\n');
        fprintf('Task=%s | output=y%d | matched demo training samples=%d\n', ...
            report.taskName,outputIndex,nTrain);
        fprintf('RNG seed=%d | ridge lambda=%.3e\n',rngSeed,ridgeLambda);
        fprintf('------------------------------------------------------------\n');
        fprintf(' index | coefficient           | basis\n');
        for k = 1:numel(coefficients)
            fprintf(' %5d | % .16e | %s\n',k,coefficients(k),basisExpressions{k});
        end
        fprintf('------------------------------------------------------------\n');
        fprintf('rank(Phi)=%d/%d | cond(Phi)=%.6e\n', ...
            matrixRank,size(PhiFit,2),conditionNumber);
        fprintf('training MSE/RMSE = %.6e / %.6e\n',trainMSE,trainRMSE);
        fprintf('Suggested Stage0SRInitialGuess:\n  ''%s''\n',guessExpression);
        fprintf('============================================================\n\n');
    end
end

function basisList = normalize_basis_list_local(value)
    if ischar(value) || (isstring(value) && isscalar(value))
        basisList = {char(value)};
    elseif isstring(value)
        basisList = cellstr(value(:));
    elseif iscell(value)
        basisList = value(:);
    else
        error('basisExpressions must be char, string, or a cell array of expressions.');
    end
    for k = 1:numel(basisList)
        basisList{k} = strtrim(char(string(basisList{k})));
        if isempty(basisList{k})
            error('basisExpressions{%d} is empty.',k);
        end
    end
end

function Phi = evaluate_basis_matrix_local(expressions,X)
    n = size(X,1);
    nx = size(X,2);
    for j = 1:nx
        eval(sprintf('x%d = X(:,%d);',j,j)); %#ok<EVLDIR>
    end
    Phi = zeros(n,numel(expressions));
    for k = 1:numel(expressions)
        expression = prepare_expression_local(expressions{k});
        validate_expression_local(expression,nx);
        try
            value = eval(expression); %#ok<EVLDIR>
        catch ME
            error('Failed to evaluate basis %d (%s): %s',k,expressions{k},ME.message);
        end
        if isscalar(value); value = repmat(value,n,1); end
        value = reshape(value,[],1);
        if numel(value) ~= n
            error('Basis %d returned %d values; expected %d.',k,numel(value),n);
        end
        Phi(:,k) = value;
    end
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

function validate_expression_local(expression,nx)
    forbidden = {';','@','[',']','{','}','''','"','=',':','!','~','\\'};
    for i = 1:numel(forbidden)
        if contains(expression,forbidden{i})
            error('Unsupported token "%s" in basis expression.',forbidden{i});
        end
    end
    allowed = {'sin','cos','tan','exp','log','sqrt','abs', ...
        'sr_square_local','sr_cube_local','sr_inv_local','sr_sqrt_abs_local', ...
        'pi','Inf','NaN','e'};
    for j = 1:nx; allowed{end+1} = sprintf('x%d',j); end %#ok<AGROW>
    residue = expression;
    for i = 1:numel(allowed)
        residue = regexprep(residue,['\<' allowed{i} '\>'],'');
    end
    residue = regexprep(residue,'[0-9eE\.\+\-\*\/\^\(\),\s]','');
    if ~isempty(residue)
        error('Unknown token(s) in basis expression: %s',residue);
    end
end

function expression = build_guess_expression_local(coefficients,bases,zeroTol,fmt)
    pieces = {};
    for k = 1:numel(coefficients)
        c = coefficients(k);
        if abs(c) <= zeroTol; continue; end
        magnitude = sprintf(fmt,abs(c));
        basis = bases{k};
        if strcmp(strrep(basis,' ',''),'1')
            term = magnitude;
        else
            term = sprintf('%s*(%s)',magnitude,basis);
        end
        if isempty(pieces)
            if c < 0; term = ['-' term]; end
            pieces{end+1} = term; %#ok<AGROW>
        else
            if c < 0
                pieces{end+1} = [' - ' term]; %#ok<AGROW>
            else
                pieces{end+1} = [' + ' term]; %#ok<AGROW>
            end
        end
    end
    if isempty(pieces); expression = '0.0'; else; expression = [pieces{:}]; end
end

function value = get_option_local(options,name,defaultValue)
    if isstruct(options) && isfield(options,name) && ~isempty(options.(name))
        value = options.(name);
    else
        value = defaultValue;
    end
end

function value = getfield_default_local(s,name,defaultValue)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
        value = s.(name);
    else
        value = defaultValue;
    end
end

function y = sr_square_local(x); y = x.^2; end
function y = sr_cube_local(x); y = x.^3; end
function y = sr_inv_local(x); y = 1./x; end
function y = sr_sqrt_abs_local(x); y = sqrt(abs(x)); end

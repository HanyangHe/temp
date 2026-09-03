function [arch, CoefExact, compileInfo] = compile_symbolic_representability_3d(task, symbolicExpressions)
%COMPILE_SYMBOLIC_REPRESENTABILITY_3D Construct the main shared 3-D PhDN DAG.
%
% Layer dimension vector: [3, 3, 3, 2, 2, 3].

    if nargin < 1 || isempty(task)
        task = task_symbolic_representability_3d();
    end
    if nargin < 2 || isempty(symbolicExpressions)
        symbolicExpressions = task.symbolicExpressions;
    end

    L = 5;
    branchActive = false(L,L);
    branchActive(1,1) = true;  % x -> h2
    branchActive(1,2) = true;  % h2 -> h3
    branchActive(1,3) = true;  % h3 -> h4
    branchActive(1,4) = true;  % h4 -> h5
    branchActive(1,5) = true;  % h5 -> output
    branchActive(5,5) = true;  % input skip -> output

    termsByBlock = cell(L,L);
    rowTerms = cell(L,L);
    rowSeedCoef = cell(L,L);

    termsByBlock{1,1} = {'1','v1^2','v2','v3','v1*v2','v2*v3'};
    rowTerms{1,1} = { ...
        {'v1*v2'}, ...
        {'1','v1^2','v2'}, ...
        {'v1^2','v3','v2*v3'} ...
    };
    rowSeedCoef{1,1} = {1, [1,1,1], [1,1,2]};

    termsByBlock{1,2} = {'exp(v1)','inv(v2)','inv(v3)'};
    rowTerms{1,2} = {{'exp(v1)'}, {'inv(v2)'}, {'inv(v3)'}};
    rowSeedCoef{1,2} = {0.25, 1, 1};

    termsByBlock{1,3} = {'v1','v2','v3'};
    rowTerms{1,3} = {{'v1','v2'}, {'v1','v3'}};
    rowSeedCoef{1,3} = {[1,1], [1,1]};

    termsByBlock{1,4} = {'log(v1)','log(v2)'};
    rowTerms{1,4} = {{'log(v1)'}, {'log(v2)'}};
    rowSeedCoef{1,4} = {1, 1};

    termsByBlock{1,5} = {'v1','v2'};
    rowTerms{1,5} = {{}, {'v1'}, {'v2'}};
    rowSeedCoef{1,5} = {[], 1, 1};

    termsByBlock{5,5} = {'cos(v2)','v1^2','sqrt(v3)','sin(v2)','v1*inv(v3)'};
    rowTerms{5,5} = { ...
        {'cos(v2)','v1^2','sqrt(v3)'}, ...
        {'sin(v2)'}, ...
        {'v1*inv(v3)'} ...
    };
    rowSeedCoef{5,5} = {[3,0.5,1], 1, 3/8};

    spec = struct();
    spec.caseLabel = 'Case 1: shared three-input/three-output operator-complete DAG';
    spec.layer = L;
    spec.hiddenDims = [3,3,2,2];
    spec.branchActiveMask = branchActive;
    spec.branchActiveMode = 'constructive_chain_plus_final_input_skip';
    spec.source = 'Manual exact compilation of the main 3-D symbolic-representability example';
    spec.termsByBlock = termsByBlock;
    spec.rowTerms = rowTerms;
    spec.rowSeedCoef = rowSeedCoef;
    spec.sharedSubexpressions = {'x1*x2', '0.25*exp(x1*x2)'};
    spec.nodeEquations = { ...
        'h2_1 = x1*x2', ...
        'h2_2 = 1+x1^2+x2', ...
        'h2_3 = x1^2+x3+2*x2*x3', ...
        'h3 = [0.25*exp(h2_1), inv(h2_2), inv(h2_3)]', ...
        'h4 = [h3_1+h3_2, h3_1+h3_3]', ...
        'h5 = [log(h4_1), log(h4_2)]', ...
        'y1 = 3*cos(x2)+0.5*x1^2+sqrt(x3)', ...
        'y2 = h5_1+sin(x2)', ...
        'y3 = h5_2+(3/8)*x1*inv(x3)' ...
    };
    spec.summary = ['The three supplied outputs are exactly represented by a shared ', ...
        'five-layer PhDN DAG with one final input skip branch.'];

    [arch, CoefExact, compileInfo] = build_manual_constructive_phdn( ...
        task, symbolicExpressions, spec);
end

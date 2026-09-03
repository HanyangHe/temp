function check_eql_official_environment(pythonExe,officialRoot)
%CHECK_EQL_OFFICIAL_ENVIRONMENT Fail early if Theano/Lasagne are unavailable.
    if nargin<1||isempty(pythonExe);pythonExe='python';end
    if nargin<2||isempty(officialRoot)
        officialRoot=fullfile(fileparts(mfilename('fullpath')),'official_eql');
    end
    mlfg=fullfile(officialRoot,'src','mlfg_final.py');
    if ~exist(mlfg,'file');error('Bundled official EQL core not found: %s',mlfg);end
    cmd=sprintf('"%s" -c "import sys,numpy,theano,lasagne; print(sys.executable); print(''EQL_OFFICIAL_ENV_OK'')"',pythonExe);
    [status,out]=system(cmd);
    if status~=0||~contains(out,'EQL_OFFICIAL_ENV_OK')
        extra='';
        if contains(out,'undefined reference to `__imp_') || contains(out,'lazylinker compiled file')
            extra=sprintf(['\nDetected a Windows Theano C-linker/import-library failure. ', ...
                'Install conda-forge libpython into eql_official, delete the stale Theano cache, ', ...
                'and rerun the environment check. See baselines/eql/EQL_WINDOWS_LINKER_REPAIR.txt.\n']);
        end
        error(['Official EQL environment check failed.\nPython: %s\n', ...
            'The bundled upstream code requires a compatible Theano/Lasagne environment.\n', ...
            'Output:\n%s%s'],pythonExe,out,extra);
    end
end

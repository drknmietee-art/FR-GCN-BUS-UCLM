function root = fr_finddata()
%FR_FINDDATA  Locate the BUS-UCLM dataset by searching upward.
%   Looks for a folder containing images/ and masks/ at, or under, each of
%   this file's ancestors.  Set FRGCN_DATAROOT in the base workspace to
%   override.
    try
        v = evalin('base', 'FRGCN_DATAROOT');
        if ischar(v) && exist(fullfile(v, 'images'), 'dir'), root = v; return; end
    catch
    end

    here = fileparts(mfilename('fullpath'));
    tried = {};
    base = here;
    for up = 0:3
        cand = {base, ...
                fullfile(base, 'BUS-UCLM Breast ultrasound lesion segmentation dataset', 'BUS-UCLM'), ...
                fullfile(base, 'BUS-UCLM')};
        % also accept any single folder here whose name starts with BUS-UCLM
        d = dir(fullfile(base, 'BUS-UCLM*'));
        for i = 1:numel(d)
            if d(i).isdir
                cand{end+1} = fullfile(base, d(i).name); %#ok<AGROW>
                cand{end+1} = fullfile(base, d(i).name, 'BUS-UCLM'); %#ok<AGROW>
            end
        end
        for i = 1:numel(cand)
            tried{end+1} = cand{i}; %#ok<AGROW>
            if exist(fullfile(cand{i}, 'images'), 'dir') && ...
               exist(fullfile(cand{i}, 'masks'),  'dir')
                root = cand{i};
                return;
            end
        end
        base = fileparts(base);
        if isempty(base), break; end
    end

    msg = sprintf(['\n\nFR-GCN could not find the BUS-UCLM dataset.\n\n' ...
        'It needs a folder containing BOTH an images/ and a masks/ subfolder,\n' ...
        'at or above:\n    %s\n\n' ...
        'Either move the dataset there, or set the path explicitly before\n' ...
        'running anything:\n\n' ...
        '    FRGCN_DATAROOT = ''C:\\path\\to\\BUS-UCLM'';\n' ...
        '    FRGCN_Main\n\nLocations checked:\n'], here);
    for i = 1:numel(tried), msg = [msg sprintf('    %s\n', tried{i})]; end %#ok<AGROW>
    error('frgcn:dataNotFound', '%s', msg);
end

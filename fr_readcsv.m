function [hdr, rows] = fr_readcsv(path)
%FR_READCSV  Minimal CSV reader (handles quoted fields containing commas).
    fid = fopen(path, 'r'); assert(fid > 0, 'cannot open %s', path);
    hdr = strsplit(strtrim(fgetl(fid)), ',');
    rows = {}; 
    while true
        ln = fgetl(fid);
        if ~ischar(ln), break; end
        if isempty(strtrim(ln)), continue; end
        parts = fr_splitcsv(ln);
        for j = 1:numel(parts)
            v = str2double(parts{j});
            if ~isnan(v) || strcmpi(strtrim(parts{j}), 'nan')
                parts{j} = v;
            end
        end
        rows(end+1, 1:numel(parts)) = parts; %#ok<AGROW>
    end
    fclose(fid);
end
function p = fr_splitcsv(ln)
    p = {}; cur = ''; inq = false;
    for i = 1:numel(ln)
        c = ln(i);
        if c == '"', inq = ~inq;
        elseif c == ',' && ~inq, p{end+1} = cur; cur = ''; %#ok<AGROW>
        else, cur(end+1) = c; %#ok<AGROW>
        end
    end
    p{end+1} = cur;
end

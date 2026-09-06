function fr_writecsv(path, header, rows)
%FR_WRITECSV  Write a cell array to CSV without any toolbox dependency.
    d = fileparts(path);
    if ~isempty(d) && ~exist(d, 'dir'), mkdir(d); end
    fid = fopen(path, 'w');
    fprintf(fid, '%s\n', strjoin(header, ','));
    for i = 1:size(rows, 1)
        parts = cell(1, size(rows, 2));
        for j = 1:size(rows, 2)
            v = rows{i, j};
            if ischar(v)
                if any(v == ','), parts{j} = ['"' v '"']; else, parts{j} = v; end
            elseif isempty(v)
                parts{j} = '';
            elseif isnan(v)
                parts{j} = 'NaN';
            elseif v == fix(v) && abs(v) < 1e9
                parts{j} = sprintf('%d', v);
            else
                parts{j} = sprintf('%.10g', v);
            end
        end
        fprintf(fid, '%s\n', strjoin(parts, ','));
    end
    fclose(fid);
end

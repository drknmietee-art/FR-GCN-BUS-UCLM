function ok = fr_need(path, figname, howto)
%FR_NEED  Report a missing input for a figure instead of erroring out.
    ok = exist(path, 'file') == 2;
    if ~ok
        fprintf('  SKIP %s -- needs %s\n       run: %s\n', ...
                figname, path, howto);
    end
end

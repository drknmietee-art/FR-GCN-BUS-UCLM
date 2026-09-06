function fr_savefig(fh, cfg, name)
%FR_SAVEFIG  Write a figure to results/figures and close it.
    p = fullfile(fr_figdir(cfg), name);
    try
        print(fh, p, '-dpng', '-r200');
    catch
        saveas(fh, p);           % fallback for toolkits without -r support
    end
    close(fh);
    fprintf('  wrote %s\n', name);
end

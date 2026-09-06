function d = fr_figdir(cfg)
%FR_FIGDIR  Figure output folder, created on demand.
    d = fullfile(cfg.outDir, 'figures');
    if ~exist(d, 'dir'), mkdir(d); end
end

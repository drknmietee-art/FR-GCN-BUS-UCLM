%FRGCN_EXTRACT  Stage 1: build and cache the superpixel graphs.
%   Runs once.  Everything cached here is independent of the fold, so the
%   cross-validation loop only has to redo the scaling, the clustering and
%   the fuzzy-rough step.
%
%   Writes cache_graphs.mat and results/dataset_manifest.csv.

cfg = fr_config();
if ~exist(cfg.outDir, 'dir'), mkdir(cfg.outDir); end

files = dir(fullfile(cfg.imgDir, '*.png'));
if isempty(files)
    error('frgcn:noImages', ['\n\nNo .png images found in\n    %s\n\n' ...
        'Check that the dataset folder contains images/ and masks/ with the\n' ...
        'original BUS-UCLM PNG files.\n'], cfg.imgDir);
end
fprintf('Dataset: %s\n', cfg.dataRoot);
fprintf('Scanning %d images in %s\n', numel(files), cfg.imgDir);

graphs = {}; manifest = {};
t0 = tic;
for i = 1:numel(files)
    name = files(i).name;
    [g, lesion, kind] = fr_loadpair(fullfile(cfg.imgDir, name), ...
                                    fullfile(cfg.maskDir, name), cfg.res);
    if ~any(lesion(:))
        continue;                       % the 419 images with no lesion
    end
    G = fr_buildgraph(g, lesion, cfg);
    G.name    = name;
    G.patient = strtok(name, '_');      % patient code is the filename prefix
    G.kind    = kind;
    G.featAll = [G.feat, fr_features_v2(G)];   % 6 base + 14 additional
    graphs{end+1} = G; %#ok<SAGROW>
    manifest(end+1, :) = {name, G.patient, kind, sum(lesion(:))}; %#ok<SAGROW>
    if mod(numel(graphs), 20) == 0
        fprintf('  %d lesion images cached (scanned %d, %.0fs)\n', ...
                numel(graphs), i, toc(t0));
    end
end
fprintf('TOTAL lesion images: %d from %d patients\n', numel(graphs), ...
        numel(unique(cellfun(@(G) {G.patient}, graphs))));

% -v7 keeps the cache readable by both MATLAB and Octave; the cache is a few
% hundred MB for the full dataset, well inside the format's 2 GB limit.
save(cfg.cacheFile, 'graphs', '-v7');
fid = fopen(fullfile(cfg.outDir, 'dataset_manifest.csv'), 'w');
fprintf(fid, 'image,patient,type,lesion_pixels\n');
for i = 1:size(manifest, 1)
    fprintf(fid, '%s,%s,%s,%d\n', manifest{i,1}, manifest{i,2}, manifest{i,3}, manifest{i,4});
end
fclose(fid);
fprintf('Wrote %s\n', cfg.cacheFile);

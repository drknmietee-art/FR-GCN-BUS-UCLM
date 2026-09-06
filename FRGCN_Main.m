%FRGCN_MAIN  Run the complete study end to end.
%
%   Fuzzy-Rough Edge Gating for Graph-Based Breast Ultrasound Lesion
%   Segmentation -- reference MATLAB implementation.
%
%   Expects the BUS-UCLM dataset one level above this folder:
%       ../BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM/
%           images/   *.png
%           masks/    *.png
%
%   Steps 1-2 and 4-6 need no toolbox at all.  Step 3 (the dense
%   convolutional baselines) needs the Deep Learning Toolbox; skip it and
%   every other table is still produced.

addpath(fileparts(mfilename('fullpath')));
t0 = tic;

fprintf('\n=== 1/6  Feature extraction ===\n');          FRGCN_Extract
fprintf('\n=== 2/6  Main experiment (5 folds x 5 seeds) ===\n'); FRGCN_RunAll

fprintf('\n=== 3/6  Dense baselines ===\n');
if exist('unetLayers','file') == 2
    FRGCN_Dense
else
    fprintf('Deep Learning Toolbox not available -- skipping dense baselines.\n');
end

fprintf('\n=== 4/6  Uncertainty study ===\n');           FRGCN_Uncertainty
fprintf('\n=== 5/6  Diagnostics ===\n');                 FRGCN_Diagnostics
fprintf('\n=== 6/6  Tables and figures ===\n');          FRGCN_Tables
FRGCN_Figures

fprintf('\nComplete in %.0f min.  Results in %s\n', toc(t0)/60, fr_config().outDir);

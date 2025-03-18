- import .stl file to blender
- choose local coordinate, use R to rotate the axis, make it uniform with the original 
- export as obj file
- transform the obj to mjcf with obj2mjcf, invoke the decompose command, reduce the number of decompose meshes

`obj2mjcf --obj-dir ./ --save-mjcf --compile-model --verbose --decompose --coacd-args.max-convex-hull 10`

-h, --help                                                                 │
│     show this help message and exit                                        │
│ --obj-dir STR                                                              │
│     path to a directory containing obj files. All obj files in the         │
│     directory will be converted (required)                                 │
│ --obj-filter {None}|STR                                                    │
│     only convert obj files matching this regex (default: None)             │
│ --save-mjcf, --no-save-mjcf                                                │
│     save an example XML (MJCF) file (default: False)                       │
│ --compile-model, --no-compile-model                                        │
│     compile the MJCF file to check for errors (default: False)             │
│ --verbose, --no-verbose                                                    │
│     print verbose output (default: False)                                  │
│ --decompose, --no-decompose                                                │
│     approximate mesh decomposition using CoACD (default: False)            │
│ --texture-resize-percent FLOAT                                             │
│     resize the texture to this percentage of the original size (default:   │
│     1.0)                                                                   │
│ --overwrite, --no-overwrite                                                │
│     overwrite previous run output (default: False)                         │
│ --add-free-joint, --no-add-free-joint                                      │
│     add a free joint to the root body (default: False)                     │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ coacd-args options ───────────────────────────────────────────────────────╮
│ arguments to pass to CoACD                                                 │
│ ────────────────────────────────────────────────────────────────────────── │
│ --coacd-args.preprocess-resolution INT                                     │
│     resolution for manifold preprocess (20~100), default = 50 (default:    │
│     50)                                                                    │
│ --coacd-args.threshold FLOAT                                               │
│     concavity threshold for terminating the decomposition (0.01~1),        │
│     default = 0.05 (default: 0.05)                                         │
│ --coacd-args.max-convex-hull INT                                           │
│     max # convex hulls in the result, -1 for no maximum limitation         │
│     (default: -1)                                                          │
│ --coacd-args.mcts-iterations INT                                           │
│     number of search iterations in MCTS (60~2000), default = 100 (default: │
│     100)                                                                   │
│ --coacd-args.mcts-max-depth INT                                            │
│     max search depth in MCTS (2~7), default = 3 (default: 3)               │
│ --coacd-args.mcts-nodes INT                                                │
│     max number of child nodes in MCTS (10~40), default = 20 (default: 20)  │
│ --coacd-args.resolution INT                                                │
│     sampling resolution for Hausdorff distance calculation (1e3~1e4),      │
│     default = 2000 (default: 2000)                                         │
│ --coacd-args.pca, --coacd-args.no-pca                                      │
│     enable PCA pre-processing, default = false (default: False)            │
│ --coacd-args.seed INT                                                      │
│     random seed used for sampling, default = 0 (default: 0)    
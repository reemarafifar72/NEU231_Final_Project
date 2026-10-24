# NEU231_Final_Project

To run the repo model --

python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

python base_model.py --dataset fmnist --dends 16 --soma 128 --sampling local

## V1-style dendritic nonlinearities

`--nonlin v1` replaces each dendrite's LeakyReLU with three fixed nonlinearities applied in
parallel (supralinear `relu(u)^p`, sublinear `s*tanh(u/s)`, calcium spike
`A*sigmoid(k*(u-theta))`). Each soma then learns a weighted sum over all three channels of
each of its own dendrites. The default `--nonlin leaky` is the paper's model, unchanged.

python base_model.py --nonlin v1 --dends 16 --soma 256            # defaults: p=2, s=1, theta=1, k=10, A=1
python base_model.py --nonlin v1 --ca-thresh 0.5 --supra-p 1.5    # change the fixed constants

Results are tagged `dann_v1_*`; `plot_results.py` draws them next to the baseline (dashed
lines, square markers). `analyze_v1.py results/dann_v1_... --data-dir data` reports which
channel each soma relies on and whether dendrites reach the Ca threshold.

## Control 1: homogeneous type ("any single nonlinearity is enough")

`--nonlin supra`, `--nonlin sub` and `--nonlin ca` give every dendrite just that one
nonlinearity, with the baseline's cable layer and parameter count. Together with the LeakyReLU
baseline they are the single-nonlinearity controls for the mixed `v1` model.

bash sweep_controls.sh                      # seeds 1-3 at 16x256 for leaky, supra, sub, ca, v1
SEEDS="1 2 3 4 5" SIZES="16x256 64x256" bash sweep_controls.sh
python plot_results.py                      # figures/7_control_homogeneous.png + control1_homogeneous.csv

The CSV compares the mixed model with the best single nonlinearity at each size and gives a
verdict: "mixed wins" if the gap exceeds two combined SDs (and 0.5 pp), else "single is enough".

# Reproducing The Best Result

1. Clone this repository and enter the project directory.
2. Download `pub.pt`, `priv.pt`, and `model.pt` from the official task release and place them in the repository root.
3. Create a Python virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy
```

4. Generate the submission file:

```bash
python guided_attack.py --mode both
```

5. The command writes `submission.csv` in the repository root.

6. To submit the file to the leaderboard, set your API key in `task_template.py` and use the provided submission code.

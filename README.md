# TinyFSL Project Setup Instructions

## Step 1: Create a New Conda Environment

Create a new conda environment with Python 3.9:

```bash
conda create --name myenv python=3.9
```

## Step 2: Initialize Conda for Bash (if needed)

If you encounter a `CommandNotFoundError` when trying to activate the environment, initialize conda for bash:

```bash
conda init bash
source ~/.bashrc
```

## Step 3: Activate the Conda Environment

Activate your new conda environment:

```bash
conda activate myenv
```

## Step 4: Install Required Python Packages

Install the required Python packages from the `requirements.txt` file:

```bash
pip install -r slt/requirements.txt
```

## Step 5: Install `freetype` if Needed

If you encounter an error related to `matplotlib` and a missing `ft2build.h` file, install `freetype`:

```bash
conda install freetype
pip install matplotlib
```

## Step 6: Install `gdown`

Install the `gdown` package to download files from Google Drive:

```bash
pip install gdown
```

## Step 7: Run the Shell Scripts

Run the necessary shell scripts to set up datasets, models, and train the model:

```bash
./datasets.sh
./models.sh
./train_model.sh
```

## Step 8: Handle Script Execution Errors

If you encounter errors running the scripts, ensure they have the correct permissions and try running them again:

```bash
chmod +rwx ./datasets.sh
chmod +rwx ./models.sh
chmod +rwx ./train_model.sh
```

# Setup CV SDK

## Install

```bash
# Create virtual environment
uv venv --python=3.12

# Install dependencies
uv sync
```

## Install OpenMMLab

```bash
task onedl:clone
task mm-install
```

## Activate environment

Activation:

```bash
# MacOS / Linux
source .venv/bin/activate

# Windows
.venv/Scripts/activate
```

Deactivation:

```bash
# MacOS / Linux
deactivate
```

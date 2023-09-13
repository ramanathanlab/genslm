# Loooooooong Sequence Lengths

<div id="fig-ds4sci">

![](../assets/deepspeed4science1.svg)

Figure 1: This work was done as part of the DeepSpeed4Science project,
in collaboration with Microsoft.

</div>

The new
[Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)
release contains a variety of improvements / optimizations to enable
pre-training Transformer based architectures with significantly longer
sequences than was previously possible.

Additional details can be found in the [📁
`DeepSpeed4Science`](https://github.com/microsoft/Megatron-DeepSpeed/examples_deepspeed/deepspeed4science/megatron_long_seq_support/README.md)
folder.

## DeepSpeed4Science (09/2023)

### New Features

- Enabled Megatron-LM’s sequence parallel.

- Enabled rotary positional embedding.

- Enabled FlashAttention v1 and v2.

- Enabled new fused kernels from NVIDIA.

### New optimizations

- Enabled attention map memory optimization, where we first generated
  attention mask on CPU memory and then moved it into GPU memory to
  avoid out-of-memory errors when training with very large sequence
  lengths.

- Position embedding partitioning, where we split weights of position
  encoding across all GPUs when enabling sequence parallel to further
  reduce the memory footprint.

## Installation

### Using [`install.sh`](https://github.com/ramanathanlab/genslm/blob/foremans/ds4sci/examples/long-sequences/install.sh)

> **Important**<br> To install, simply:
>
> ``` bash
> git clone https://github.com/ramanthanlab/GenSLM/
> cd GenSLM/examples/long-sequences/
> ./install.sh
> ```
>
> Explicitly,
> [`./install.sh`](https://github.com/ramanathanlab/genslm/blob/foremans/ds4sci/examples/long-sequences/install.sh)
> will:
>
> 1.  **Automatically** create a virtual environment *on top of* the
>     latest `conda` module
> 2.  Install (+ update[^1]) / build all the required
>     [dependencies](#dependencies) into this virtual environment

### Step-by-Step

For completeness, we describe below the steps for installing and
building each of the dependencies:

1.  Clone GitHub repo:

    ``` bash
    git clone https://github.com/ramanthanlab/GenSLM
    ```

2.  Load `conda` module:

    - ThetaGPU:

      ``` bash
      # ThetaGPU:
      export MACHINE="ThetaGPU"
      export CONDA_DATE="2023-01-10"
      module load conda/2023-01-11
      conda activate base
      ```

    - Polaris:

      ``` bash
      # Polaris:
      export MACHINE="Polaris"
      export CONDA_DATE="2023-01-10"
      module load conda/2023-01-10-unstable
      conda activate base
      ```

3.  Setup Virtual Environment:

    ``` bash
    # create a new virtual environment
    cd Megatron-DeepSpeed
    python3 -m venv "venvs/${MACHINE}/${CONDA_DATE}" --system-site-packages
    source "venvs/${MACHINE}/${CONDA_DATE}/bin/activate"
    ```

#### Dependencies

1.  [ `saforem2/ezpz`](https://github.com/saforem2/ezpz)

    ``` bash
    pip install -e "git+https://github.com/saforem2/ezpz.git#egg=ezpz"
    ```

2.  [ `Microsoft/DeepSpeed`](https://github.com/microsoft/DeepSpeed)

    ``` bash
    git clone https://github.com/microsoft/DeepSpeed.git
    cd DeepSpeed
    python3 -m pip install -e .
    ```

3.  [
    `Microsoft/Megatron-DeepSpeed`](https://github.com/microsoft/Megatron-DeepSpeed):

    ``` bash
    git clone https://github.com/microsoft/Megatron-DeepSpeed.git
    ```

4.  [ `NVIDIA/apex`](https://github.com/NVIDIA/apex)

    ``` bash
    git clone https://github.com/NVIDIA/apex
    cd ../apex/
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" -e ./
    ```

5.  [ `pybind/PyBind11`](https://github.com/pybind/pybind11)

    ``` bash
    pip install pybind11
    ```

6.  [
    `Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention):

    <div>

    > **Flash Attention**
    >
    > - The new release supports three different implementations of
    >   FlashAttention: (`v1.0.4`, `v2.x`, `triton`)
    > - FlashAttention `v2.x` may have numerical instability issues. For
    >   the best performance, we recommend using FlashAttention + Triton

    </div>

    - `v1.0.4`:

      ``` bash
      python3 -m pip install flash-attn==1.0.4
      ```

    - `v2.x`:

      ``` bash
      git clone https://github.com/Dao-AILab/flash-attention
      cd flash-attention
      python3 setup.py install
      ```

    - `openai/triton`:

      ``` bash
      git clone -b legacy-backend https://github.com/openai/triton
      cd triton/python
      python3 -m pip install cmake
      python3 -m pip install .
      ```

### Running

The [`ALCF/`](./ALCF/) directory contains shell scripts for setting up
the environment and specifying the options to be used when launching.

Various options can be specified dynamically at runtime by setting them
in your environment, e.g.:

``` bash
MODEL_SIZE_KEY="GPT25B" SEQ_LEN=128000 USE_FLASH_ATTN=1 MICRO_BATCH=1 GAS=1 SP_TYPE="megatron" ZERO_STAGE=1 ./ALCF/train-gpt3.sh
```

Explicitly:

- [`ALCF/train-gpt3.sh`](./ALCF/train-gpt3.sh): **Main entry point for
  training**
  - This script will **automatically** source the rest of the required
    [`ALCF/*.sh`](./ALCF/) scripts below
- [`ALCF/models.sh`](./ALCF/models.sh): Contains some example model
  architectures for GPT3-style models
- [`ALCF/args.sh`](./ALCF/args.sh): Logic for parsing / setting up
  runtime options for Megatron and DeepSpeed
- [`ALCF/setup.sh`](./ALCF/args.sh): Locate and activate virtual
  environment to be used, ensure MPI variables are set properly
- [`ALCF/launch.sh`](./ALCF/launch.sh): Identify available resources and
  build the command to be executed
  - i.e. figure out how many: `{nodes, GPUs per node, GPUs total}`, to
    pass to `mpi{run,exec}`
  - then, use this to build
    `mpiexec <mpiexec-args> python3 pretrain_gpt.py`

## Initial Results

<div>

> **PRE-RELEASE**
>
> I’ve kept in the (executable) code blocks for the time being (just to
> show how I’m generating the bar plots in [Figure 2](#fig-seq-len)) but
> these can be ommitted in the actual README

</div>

<details>
<summary>Code</summary>

``` python
%matplotlib inline
import matplotlib_inline
import os
import numpy as np
import datetime
from typing import Tuple
import matplotlib.pyplot as plt
from pathlib import Path
# NOTE:
# - [Toolbox](https://github.com/saforem2/toolbox)
from toolbox import set_plot_style
import seaborn as sns
from opinionated import STYLES
import seaborn as sns

sns.set_context('talk')
set_plot_style()
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

plt.style.use('default')
set_plot_style()
plt.style.use(STYLES['opinionated_min'])
plt.rcParams['ytick.labelsize'] = 14.0
plt.rcParams['xtick.labelsize'] = 14.0
plt.rcParams['grid.alpha'] = 0.4

grid_color = plt.rcParams['grid.color']

def save_figure(
        fname: str,
        outdir: os.PathLike,
):
    pngdir = Path(outdir).joinpath('pngs')
    svgdir = Path(outdir).joinpath('svgs')
    pngdir.mkdir(exist_ok=True, parents=True)
    svgdir.mkdir(exist_ok=True, parents=True)
    pngfile = pngdir.joinpath(f'{fname}.png')
    svgfile = svgdir.joinpath(f'{fname}.svg')
    _ = plt.savefig(pngfile, dpi=400, bbox_inches='tight')
    _ = plt.savefig(svgfile, dpi=400, bbox_inches='tight')
```

</details>

    Failed to download font: Source Sans Pro, skipping!
    Failed to download font: Titillium WebRoboto Condensed, skipping!

<details>
<summary>Data</summary>

``` python
gpus = ('32', '64', '128')

colors = {
    'Old': '#007DFF',
    'Megatron-LM': '#76b900',
    'Megatron-DS':  '#FF5252',
}

data = {
    '25B': {
        'Old': np.array([28, 32, 32]),
        'Megatron-LM': np.array([14, 46, 52]),
        'Megatron-DS': np.array([128, 384, 448]),
    },
    '33B': {
        'Old': np.array([36, 42, 42]),
        'Megatron-LM': np.array([26, 48, 52]),
        'Megatron-DS': np.array([192, 448, 512]),
    }
}
```

</details>

<div id="fig-seq-len" style="background-color:none;">

<details>
<summary>Make the plots</summary>

``` python
x = np.arange(len(gpus))
width = 0.25
multiplier = 0

outdir = Path(os.getcwd()).joinpath('assets')
outdir.mkdir(exist_ok=True, parents=True)

improvement = {}
for idx, (model_size, d) in enumerate(data.items()):
    multiplier = 0
    figure, axes = plt.subplots(figsize=(6.4, 4.8))
    fig = plt.gcf()
    ax = plt.gca()
    for label, value in d.items():
        offset = width * multiplier
        rects = ax.barh(
          x + offset,
          value,
          width,
          label=label,
          color=colors[label],
          alpha=0.8
        )
        ax.bar_label(
          rects,
          padding=3,
          color=colors[label],
          family='monospace',
          weight='bold'
        )
        multiplier += 1
    ax.set_ylabel(
        'GPUs',
        fontsize=18,
        family='sans-serif',
        loc='center',
    )
    ax.set_yticks(x + width, gpus)
    plt.figtext(
        0.00, 0.94, f"{model_size}", fontsize=24, fontweight='bold', ha='left'
    )
    ax.set_xlabel(
        'Sequence Length (k)', fontsize=18, loc='center'
    )
    ax.legend(
        bbox_to_anchor=(0.005, 1.05, 0.99, .098),
        alignment='center',
        edgecolor="#83838320",
        frameon=True,
        ncols=3,
        fontsize=14,
        mode="expand",
        borderaxespad=0.01
    )
    save_figure(fname=f'{model_size}', outdir=outdir)
    _ = plt.show()
```

</details>

![GPT-`25B` Model](dsblog_files/figure-commonmark/cell-4-output-1.svg)

![GPT-`33B` Model](dsblog_files/figure-commonmark/cell-4-output-2.svg)

Figure 2: Pre-training with long sequence support across different model
sizes and numbers of GPUs. In each case, the `new` (current)
implementation **significantly** outperforms both NVIDIA/Megatron-LM as
well as our previous implementation.

</div>

<div id="fig-table">

| Sequence Length |     Old Megatron-DeepSpeed (TFLOPS)      |      New Megatron-DeepSpeed (TFLOPS)      |
|:---------------:|:----------------------------------------:|:-----------------------------------------:|
|       2k        | <span style="text-weight:600;">25</span> | <span style="text-weight:600;">68</span>  |
|       4k        | <span style="text-weight:600;">28</span> | <span style="text-weight:600;">80</span>  |
|       8k        |    <span class="red-text">OOM</span>     | <span style="text-weight:600;">86</span>  |
|       16k       |    <span class="red-text">OOM</span>     | <span style="text-weight:600;">92</span>  |
|       32k       |    <span class="red-text">OOM</span>     | <span style="text-weight:600;">100</span> |
|       64k       |    <span class="red-text">OOM</span>     | <span style="text-weight:600;">106</span> |
|      128k       |    <span class="red-text">OOM</span>     | <span style="text-weight:600;">119</span> |
|      256k       |    <span class="red-text">OOM</span>     | <span style="text-weight:600;">94</span>  |

The following experiments are performed on 4 NVIDIA DGX A100-40GB nodes,
all using `TPSIZE=32`, connected through 8 HDR InfiniBand (200Gb/s per
HDR).

Figure 3: TP stands for tensor parallelism.

</div>

## ZeRO Offloading

These newly introduced optimizations, in combination with
[ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/) allows
us to go even further.

By employing ZeRO-Offloading, we are able to free up additional memory
which can be used for *even longer* sequences.

Though work is still ongoing, this is a promising direction that will
allow us to consider significantly larger genomes than previously
possible.

<div id="fig-wandb">

<div style="padding:0.5rem; border: 1px solid var(--dim-text); border-radius: 0.2rem;">

<iframe src="https://wandb.ai/l2hmc-qcd/Megatron-DS-Benchmarking/reports/Looooooong-Sequences--Vmlldzo1MzI2NjA1" style="border:none;height:1024px;width:100%">
</iframe>

</div>

Figure 4: Weights & Biases Report

</div>

[^1]:

    2.  `deepspeed-0.10.3`
    3.  `pytorch==2.0.0+cu118`

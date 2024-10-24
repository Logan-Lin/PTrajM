# PTrajM: Efficient and Semantic-rich Trajectory Learning with Pretrained Trajectory-Mamba

Implementation code of the *Pretrained Trajectory-Mamba* (**PTrajM**) model.

Preprint paper: https://arxiv.org/abs/2408.04916

## Hands-on

Set OS env parameters:

```bash
export SETTINGS_CACHE_DIR=/dir/to/cache/setting/files;
export MODEL_CACHE_DIR=/dir/to/cache/model/parameters;
export PRED_SAVE_DIR=/dir/to/save/predictions;
```

Run the main script:

```bash
python main.py -s local_test;
```

## Model Structure

![PTrajM-traj-mamba](./assets/PTrajM-traj-mamba.png)

The learnable model of PTrajM, Trajectory-Mamba. It incorporates movement behavior parameterization and a trajectory state-space model (Traj-SSM) to extract continuous movement behavior.

![PTrajM-pretrain](./assets/PTrajM-pretrain.png)

The travel purpose-aware pre-training procedure of PTrajM. It aligns the learned embeddings of Trajectory-Mamba with the travel purpose identified by the road and POI encoders.

## Technical Structure

The parameters and experimental settings are controlled by a JSON configuration file. `settings/local_test.json` provides an example.

### Code Overview

The project includes several key components:

1. **Mamba Model**: The core Trajectory-Mamba model is implemented in the `models/mamba2/ssd_combined.py` file.

2. **Data Processing**: The `sample` directory contains subsets of the Chengdu and Xian datasets for reference and quick debugging. These can be used to test the model's functionality before working with the full datasets. Preprocess code is included in the `data.py` file.

3. **Configuration**: The project uses JSON files for configuration. You can find an example in `settings/local_test.json`.

4. **Main Script**: The `main.py` script is the entry point for running experiments and training the model.

### Dataset

The `sample` directory contains subsets of the Chengdu and Xian datasets for reference and quick debugging. The full datasets have the same file format and fields. These datasets include trajectory data that can be used to train and evaluate the PTrajM model.

## Contact

If you have any further questions or would like to request the full datasets, feel free to contact me directly. My contact information is available on my homepage: https://www.yanlincs.com/
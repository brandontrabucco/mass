![MaSS: 3D Mapping & Semantic Search](images/teaser.png)

# MaSS: 3D Mapping & Semantic Search

This repository is the official implementation of the [paper](https://openreview.net/forum?id=1C6nCCaRe6p) "A Simple Approach For Visual Room Rearrangement: 3D Mapping & Semantic Search" at ICLR 2023. Our method won the 2022 Rearrangement Challenge at the Embodied AI Workshop, and serves as a baseline for the 2023 challenge. 

## Installation

You can install MaSS by installing required packages with `pip install -r requirements.txt`, following the installation instructions for the [ai2thor-rearrangement package](https://github.com/allenai/ai2thor-rearrangement) on GitHub, and then [installing detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

Once these are installed, MaSS can be installed via `pip install -e .`.

Our method was developed using PyTorch 1.10.2. Newer versions may be compatible, but are untested.

Two model checkpoints are required to run MaSS: 

(1) a trained Mask R-CNN checkpoint compatible with `detectron2` available [here](https://drive.google.com/drive/folders/1IpwdFjlb5B4oStDfjyI8cho3nJpnpVPw?usp=share_link).
(2) a Semantic Search policy that was downloaded with the cloned repository, named `policy.pth` in the same directory as this README.

## Running The Agent

Following the instructions [here](https://github.com/allenai/ai2thor-rearrangement) first export the challenge package to your `PYTHONPATH`.

```bash
export PYTHONPATH=$PYTHONPATH::/path/to/ai2thor-rearrangement
```

Then you can run the agent by calling agent.py with your python environment.

```bash
python -u agent.py \
--logdir ./testing-the-agent --stage val \
--semantic-search-walkthrough \
--semantic-search-unshuffle \
--use-feature-matching \
--start-task 0 --total-tasks 20
```

The above command runs MaSS using our Semantic Search policy to select navigation goals during the walkthrough phase and the unshuffle phase. In addition, the `--use-feature-matching` option uses image features to match instances of objects between the unshuffle phase and walkthrough phase, and should be used as it improves `%FixedStrict` by 7.03 points in our experiments.

## Citation

If you find our work helpful in your research, consider citing our paper at ICLR 2023:

```
@inproceedings{
    trabucco2023a,
    title={A Simple Approach for Visual Room Rearrangement: 3D Mapping and Semantic Search},
    author={Brandon Trabucco and Gunnar A Sigurdsson and Robinson Piramuthu and Gaurav S. Sukhatme and Ruslan Salakhutdinov},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=1C6nCCaRe6p}
}
```

In addition, consider citing the Rearrangement Challenge benchmark:

```
@InProceedings{RoomR,
  author = {Luca Weihs and Matt Deitke and Aniruddha Kembhavi and Roozbeh Mottaghi},
  title = {Visual Room Rearrangement},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2021}
}
```

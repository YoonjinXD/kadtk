# Kernel Audio Distance Toolkit
The Kernel Audio Distance Toolkit (KADTK) provides an efficient and standardized implementation of Kernel Audio Distance (KAD)—a distribution-free, unbiased, and computationally efficient metric for evaluating generative audio.

## 1. Installation

To use the KAD toolkit, you must first install it. This library is created and tested on Python 3.10 on Linux but should work on Python >3.9.

### 1.1 Install
To install the required libraries, run:
```sh
poetry install
```

If you gets CUDA related error, ensure your device is GPU-compatible and install the necessary software for CUDA support.
Only pytorch~=2.1.x officially supported.

### 1.2 Troubleshooting
- if scipy causes some error, reinstall scipy: pip uninstall scipy && pip install scipy==1.11.2
- if charset causes some error, (re)install chardet: pip install chardet
- if CUDA causes some error, ensure your device is GPU-compatible and install the necessary software for CUDA support. Only pytorch~=2.1.x officially supported.


## 2. Usage
The toolkit provides a CLI command for computing KAD scores. It automatically extracts target embeddings and computes the KAD score between your generated audio files and an evaluation dataset.
```sh
kadtk {model_name} {target-set dir} {evaluation-set dir}
```

(Enable Options)

*--fad* compute Fréchet Audio Distance instead of Kernel Audio Distance. <br/>
*--inf* option uses metric-inf extrapolation, and *--indiv* calculates metric for individual audios. <br/>
*--force-emb-encode* forces re-extraction of embeddings, not using cache. <br/>
*--force_stats-calc* forces re-calculation of kernel statistics, not using cache. <br/>


(Examples)
```sh
kadtk panns-wavegram-logmel /test_samples/generated_samples /test_samples/dcase2023_eval
kadtk vggish /test_samples/generated_samples /test_samples/dcase2023_eval --fad # it will calculate FAD instead of KAD
kadtk passt-fsd50k /test_samples/generated_samples /test_samples/dcase2023_eval --indiv scores.csv # it will save indivisual sample-wise scores in scores.csv
kadtk-embeds -m vggish panns-wavegram-logmel -d /test_samples/generated_samples /test_samples/dcase2023_eval # will only save each embeddings

```

## 3. Supported Models

| Model | Name in KADtk | Description | Creator |
| --- | --- | --- | --- |
| [CLAP](https://github.com/microsoft/CLAP) | `clap-2023` | general audio representation | Microsoft |
| [CLAP](https://github.com/LAION-AI/CLAP) | `clap-laion-{audio/music}` | general audio, music representation | LAION |
| [MERT](https://huggingface.co/m-a-p/MERT-v1-95M) | `MERT-v1-95M-{layer}` | music understanding | m-a-p |
| [VGGish](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md) | `vggish` | general audio embedding | Google |
| [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn/README.md) | `panns-cnn14-{16k/32k}, panns-wavegram-logmel` | general audio embedding | Kong, Qiuqiang, et al. |
| [OpenL3](https://github.com/marl/openl3/README.md) | `openl3-{mel256/mel128}-{env/music}` | general audio embedding | Cramer, Aurora et al. |
| [PaSST](https://github.com/kkoutini/passt_hear21/README.md) | `passt-{base-{10s/20s/30s}, passt-openmic, passt-fsd50k` (10s default, base for AudioSet) | general audio embedding | Koutini, Khaled et al. |
| [Encodec](https://github.com/facebookresearch/encodec) | `encodec-emb` | audio codec | Facebook/Meta Research |
| [DAC](https://github.com/descriptinc/descript-audio-codec) | `dac-44kHz` | audio codec | Descript |
| [CDPAM](https://github.com/pranaymanocha/PerceptualAudio) | `cdpam-{acoustic/content}` | perceptual audio metric | Pranay Manocha et al. |
| [Wav2vec 2.0](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md) | `w2v2-{base/large}` | speech representation | Facebook/Meta Research |
| [HuBERT](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md) | `hubert-{base/large}` | speech representation | Facebook/Meta Research |
| [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) | `wavlm-{base/base-plus/large}` | speech representation | Microsoft |
| [Whisper](https://github.com/openai/whisper) | `whisper-{tiny/base/small/medium/large}` | speech recognition | OpenAI |


### Optional Dependencies

Optionally, you can install dependencies that add additional embedding support. They are:

* PANNS(updated): `pip install git+https://github.com/DCASE2024-Task7-Sound-Scene-Synthesis/fadtk.git`
* CDPAM: `pip install cdpam`
* DAC: `pip install descript-audio-codec==1.0.0`


## 4. Citation, Acknowledgments and Licenses

```latex
@article{fad_embeddings,
    author = {Tailleur, Modan and Lee, Junwon and Lagrange, Mathieu and Choi, Keunwoo and Heller, Laurie M. and Imoto, Keisuke and Okamoto, Yuki},
    title = {Correlation of Fréchet Audio Distance With Human Perception of Environmental Audio Is Embedding Dependant},
    journal = {arXiv:2403.17508},
    url = {https://arxiv.org/abs/2403.17508},
    year = {2024}
}
```

```latex
@inproceedings{fadtk,
  title = {Adapting Frechet Audio Distance for Generative Music Evaluation},
  author = {Azalea Gui, Hannes Gamper, Sebastian Braun, Dimitra Emmanouilidou},
  booktitle = {Proc. IEEE ICASSP 2024},
  year = {2024},
  url = {https://arxiv.org/abs/2311.01616},
}
```
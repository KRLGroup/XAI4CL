![alt text](https://github.com/KRLGroup/XAI4CL/blob/main/logo.jpg)

This repository re-implements existing XAI-guided continual learning methods, and allows testing them on different datasets and scenarios to provide ready-to-use baselines. It is still a work-in-progress, and we welcome contributions from the community to turn our initial efforts into a library.

## XAI-guided Continual Learning Approaches
The following table summarizes existing XAI-guided continual learning approaches, providing useful references.
| Name                                               | Abbreviation | Reference             | Venue                | Github                                                          |
|----------------------------------------------------|--------------|-----------------------|----------------------|---------------------------------------------------------------------------------------|
| Learning without Memorizing                        | LwM          | [Dhar, P. et al. (2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dhar_Learning_Without_Memorizing_CVPR_2019_paper.pdf)           | CVPR                 | [Link](https://github.com/stony-hub/learning_without_memorizing)                             |
| Remembering for the Right Reasons                  | RRR          | [Ebrahimi, S. et al. (2021)](https://openreview.net/pdf?id=tHgJoMfy6nI)           | ICLR                 | [Link](https://github.com/SaynaEbrahimi/Remembering-for-the-Right-Reasons)                    |
| Adversarial Shapley value Experience Replay        | ASER         | [Shim, D. et al. (2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17159)          | AAAI                 | [Link](https://github.com/RaptorMai/online-continual-learning)                                |
| Semi-Quantized Activation Neural Networks          | SQANN        | [Tjoa, E. et al. (2022)](https://openreview.net/forum?id=xOHuV8s7Yl)         | ICLR Reject                | [Link](https://github.com/ericotjo001/explainable_ai)                                         |
| Relevance-based Neural Freezing                    | RNF          | [Ede, S. et al. (2022)](https://link.springer.com/chapter/10.1007/978-3-031-14463-9_1)           | CD-MAKE              | -                                                                                     |
| Dual View Consistency                              | DVC          | [Gu, Y. et al. (2022)](https://ieeexplore.ieee.org/abstract/document/9879220)           | CVPR                 | [Link](https://github.com/YananGu/DVC)                                                        |
| Experience Packing and Replay                      | EPR          | [Saha, G. et al. (2023)](https://openaccess.thecvf.com/content/WACV2023/html/Saha_Saliency_Guided_Experience_Packing_for_Replay_in_Continual_Learning_WACV_2023_paper.html)           | WACV                 | -                                                                                     |
| XAI-Increment                                      | XAI-I        | [Mazumder, A.N. et al. (2023)](https://www.researchgate.net/publication/365081375_XAI-Increment_A_Novel_Approach_Leveraging_LIME_Explanations_for_Improved_Incremental_Learning)| EUSIPCO              | -                                                                                     |
| Interpretable Class-InCremental\newline LEarning   | ICICLE       | [D. Rymarczyk, J. et al. (2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Rymarczyk_ICICLE_Interpretable_Class_Incremental_Continual_Learning_ICCV_2023_paper.pdf)        | ICCV                 | [Link](https://github.com/gmum/ICICLE)                                               |
| Shape and Semantics-based Selective Regularization | S3R          | [Zhang, J. et al. (2023)](https://ieeexplore.ieee.org/document/10078916)           | IEEE Medical Imaging | [Link](https://github.com/jingyzhang/S3R)                                                     |
| Saliency-Augmented Memory Completion               | SAMC         | [Bai, G. et al. (2023)](https://epubs.siam.org/doi/pdf/10.1137/1.9781611977653.ch28)          | SDM                  | [Link](https://github.com/BaiTheBest/SAMC)                                                    |
| Concept Controller                                 | CC           | [Yang, S. et al. (2024)](https://openreview.net/forum?id=pGL4P2kg6V&noteId=vPp16Pn9BE)            | ICLR Reject          | -                                                                                     |

Up to now, our repository reimplements RRR and EPR.

## Environment Set-up
We provide a ready-to-use environment to perform experiments, following these steps:
* Download and install docker following the steps at [this link](https://docs.docker.com/engine/install/).
* Pull the PyTorch docker image optimized by NVIDIA
  ```docker pull nvcr.io/nvidia/pytorch:23.12-py3```
* Clone this repo
  ```git clone https://github.com/KRLGroup/XAI4CL.git```
* Move inside the Dockerfile directory
  ```cd Dockerfile/```
* Build the custom image
  ```docker build -t xai4cl:1.0 .```
* Move to the parent directory
  ```cd ..```
* Run the docker container
  ```docker run --gpus all -it --rm -v XAI4CL:/workspace/ xai4cl:1.0```

## XAI4CL Implementation
XAI4CL is built on top of Avalanche, a modular and extensible PyTorch-based library tailored for CL, developed with a focus on reproducibility, scalability, and ease of experimentation. It is structured around five core modules: \textit{benchmarks}, \textit{training}, \textit{models}, \textit{evaluation}, and \textit{logging}, each supporting different stages of a CL pipeline.

The _benchmark_ module facilitates the definition and manipulation of CL scenarios. It provides both standard and custom benchmarks by organizing data into streams and experiences, representing sequential learning tasks. This enables the flexible simulation of scenarios like TIL, CIL, and DIL learning.

The _training_ module offers a suite of predefined strategies and supports the construction of hybrid approaches by combining multiple techniques. Central to this design is a plugin-based architecture, which allows researchers to inject additional behavior into training loops without modifying core strategy implementations. In this context, we integrate XAI-guided CL methods as \textit{strategy plugins} implementing attribution-based regularizers, replay buffers with saliency selection, or other explanation-driven components which can be used with the \textit{Naive} baseline (i.e., standard sequential training), in combination with other compatible strategies (e.g., EWC).

The _model_ module introduces support for dynamic architectures, allowing the network structure to evolve over time, a key requirement in lifelong learning. Multi-head classifiers, progressive networks, and other adaptive architectures are readily supported.

For performance tracking, the _evaluation_ module includes an extensive set of metrics, ranging from accuracy to memory usage, and supports both standalone and plugin-based usage. These metrics can be visualized or stored using the _logging_ module.

Finally, Avalanche's design philosophy emphasizes composability. Strategies are built atop reusable templates, and the plugin interface allows XAI components to seamlessly interact with internal states of the learning process, such as modifying the loss function before each update or altering the data stream based on explanation scores. This flexibility has been key in enabling our unified implementation of XAI-guided CL methods within the \texttt{XAI4CL} repository.

## Example Usage
To run an experiment using one of our predifined configuration files, execute the following command from the _src_ folder:

```python main.py --config ../configs/<config_name>.yaml```

e.g.

```python main.py --config ../configs/rrr_cifar10.yaml```

This command will automatically run the chosen experiments using 3 different random seeds.

By running experiments using our _main.py_ with predefined configuration files, the _MetricsCheckpoint_ plugin we implement in the _plugins_ folder is included, and therefore all metrics are automatically stored and plotted during training.

## Tutorials
We also provide a notebook in the _tutorials_ folder, in which we show step by step how to load a benchmarck, define a model, a base strategy, and add any of our plugins to test the desired CL strategy. For any information regarding benchmarks, models, and evaluation, please refer to Avalanche documentation.

## Citation
If you use this code, please cite:
```
@article{https://doi.org/10.1002/widm.70046,
  author = {Proietti, Michela and Ragno, Alessio and Capobianco, Roberto},
  title = {XAI-Guided Continual Learning: Rationale, Methods, and Future Directions},
  journal = {WIREs Data Mining and Knowledge Discovery},
  volume = {15},
  number = {4},
  pages = {e70046},
  keywords = {continual learning, explainable artificial intelligence, explanation guided learning},
  doi = {https://doi.org/10.1002/widm.70046},
  url = {https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/widm.70046},
  eprint = {https://wires.onlinelibrary.wiley.com/doi/pdf/10.1002/widm.70046},
  note = {e70046 DMKD-00697.R2},
  year = {2025}
}
```

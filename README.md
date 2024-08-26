# XAI4CL

*Under construction...*

This is the official code repository of *XAI-Guided Continual Learning: Rationale, Methods, and Future Directions*. It re-implements existing XAI-guided continual learning methods, testing them on different datasets and scenarios to provide ready-to-use baselines.

## XAI-Guided Continual Learning: Rationale, Methods, and Future Directions
### Abstract
Providing neural networks with the ability to learn new tasks sequentially represents one of the main challenges in artificial intelligence. Indeed, neural networks are prone to losing previously acquired knowledge upon learning new information, a phenomenon known as catastrophic forgetting. Continual learning proposes diverse solutions to mitigate this problem, but only a few leverage explainable artificial intelligence. This work justifies using explainability techniques in continual learning, emphasizing the need for greater transparency and trustworthiness in these systems and identifying a neuroscientific rationale in the similarities between the forgetting mechanisms in biological and artificial neural networks. Finally, we review existing research applying explainability methods to address catastrophic forgetting, organizing them into a comprehensive taxonomy and proposing potential avenues for future research on this topic.

### XAI-guided Continual Learning
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

## Environment set-up
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
  ```docker run --gpus all -it --rm -v XAI4CL:/workspace/ --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --env="DISPLAY" --net=host --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host xai4cl:1.0```

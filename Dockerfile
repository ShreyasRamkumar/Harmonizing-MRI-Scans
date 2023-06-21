FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
LABEL Author="Shreyas Ramkumar"

# Install base utilities
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update 
RUN apt-get install software-properties-common -y 
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    add-apt-repository ppa:deadsnakes/ppa


RUN apt-get update && apt-get -y install git && \
    apt-get -y install python3.9 && \
    apt-get install -y python3-pip && \
    apt-get install -y python-is-python3
RUN python --version
# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

#SHELL ["/bin/bash", "--login", "-c"]
COPY environment.yml environment.yml
RUN set -x \
    && conda init bash \
    && . /root/.bashrc \
    && conda activate base

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip install pandas \
    einops jupyterlab matplotlib \ 
    networkx plotly scikit-learn \
    shap imbalanced-learn 
RUN pip install tensorboard \
    openpyxl xgboost xlrd nibabel \
    nibabel markdown pdoc3 \
    pytorch-lightning "ray[default]" \
    treelib monai torchio "ray[tune]" 
RUN pip install GPUtil batchgenerators \
    hydra-core pyrsistent hydra-ray-launcher \
    wandb \
    torchmetrics
RUN pip install dipy numpy pyrobex simpleitk
RUN pip install git+https://github.com/eduardojdiniz/radio
RUN . /root/.bashrc
ENV PATH /opt/conda/bin:$PATH

COPY Preprocessing.py Preprocessing.py
COPY network_tl.py network_tl.py 
RUN chmod +x network_tl.py
RUN chmod +x Preprocessing.py

#RUN conda env update -qf environment.yml

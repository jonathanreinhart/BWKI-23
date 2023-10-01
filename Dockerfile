FROM python:3.11

WORKDIR /app

COPY KI KI

RUN pip install numpy
RUN pip install mne
RUN pip install matplotlib
RUN pip install torch
RUN pip install scikit-learn
RUN pip install einops
RUN pip install typing
RUN pip install torchinfo
RUN pip install tensorboard
RUN pip install timm

CMD ["python", "./KI/Inference.py"]

FROM rocm/tensorflow:rocm7.2.4-py3.10-tf2.19-dev

RUN pip install jupyter matplotlib scikit-learn seaborn pillow tqdm japanize-matplotlib && \
    pip install medmnist --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /workspace
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

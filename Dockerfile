FROM icepack/firedrake-python3.8:0.5.7

MAINTAINER shapero.daniel@gmail.com

RUN sudo apt-get update && \
    DEBIAN_FRONTEND="noninteractive" sudo apt-get install -yq \
    ffmpeg \
    graphviz \
    libgraphviz-dev

# Disable the IPython history when running as a Jupyter kernel. This prevents
# all sorts of annoying warnings associated to the history db getting locked.
RUN mkdir -p /home/sermilik/.ipython/profile_default/ && \
    echo "c = get_config(); c.HistoryManager.enabled = False" >> /home/sermilik/.ipython/profile_default/ipython_kernel_config.py

# Install the github actions runner.
ARG token

RUN mkdir actions-runner && \
    cd actions-runner && \
    curl -o actions-runner-linux-x64-2.287.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.287.1/actions-runner-linux-x64-2.287.1.tar.gz && \
    echo "8fa64384d6fdb764797503cf9885e01273179079cf837bfc2b298b1a8fd01d52  actions-runner-linux-x64-2.287.1.tar.gz" | shasum -a 256 -c && \
    tar xzf ./actions-runner-linux-x64-2.287.1.tar.gz && \
    ./config.sh --url https://github.com/danshapero/danshapero.github.io --token ${token} --unattended

WORKDIR /home/sermilik/actions-runner
ENTRYPOINT bin/runsvc.sh

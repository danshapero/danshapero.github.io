FROM icepack/firedrake-python3.8:0.5.6

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
    curl -o actions-runner-linux-x64-2.286.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.286.1/actions-runner-linux-x64-2.286.1.tar.gz && \
    echo "7b1509c353ea4e6561b2ed2e916dcbf2a0d8ce21172edd9f8c846a20b6462cd6  actions-runner-linux-x64-2.286.1.tar.gz" | shasum -a 256 -c && \
    tar xzf ./actions-runner-linux-x64-2.286.1.tar.gz && \
    ./config.sh --url https://github.com/danshapero/danshapero.github.io --token ${token} --unattended

ENTRYPOINT actions-runner/run.sh

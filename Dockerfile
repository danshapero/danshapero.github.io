FROM firedrakeproject/firedrake-vanilla:2025-01

LABEL org.opencontainers.image.authors="shapero.daniel@gmail.com"

RUN sudo apt-get update && \
    DEBIAN_FRONTEND="noninteractive" sudo apt-get install -yq \
    ffmpeg \
    graphviz \
    libgraphviz-dev

# Disable the IPython history when running as a Jupyter kernel. This prevents
# all sorts of annoying warnings associated to the history db getting locked.
RUN mkdir -p /home/firedrake/.ipython/profile_default/ && \
    echo "c = get_config(); c.HistoryManager.enabled = False" >> /home/firedrake/.ipython/profile_default/ipython_kernel_config.py

# Install the github actions runner.
ARG token

RUN mkdir actions-runner && \
    cd actions-runner && \
    curl -o actions-runner-linux-x64-2.323.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.323.0/actions-runner-linux-x64-2.323.0.tar.gz && \
    echo "0dbc9bf5a58620fc52cb6cc0448abcca964a8d74b5f39773b7afcad9ab691e19  actions-runner-linux-x64-2.323.0.tar.gz" | shasum -a 256 -c && \
    tar xzf ./actions-runner-linux-x64-2.323.0.tar.gz && \
    ./config.sh --url https://github.com/danshapero/danshapero.github.io --token ${token} --unattended

WORKDIR /home/firedrake/actions-runner
ENTRYPOINT bin/runsvc.sh

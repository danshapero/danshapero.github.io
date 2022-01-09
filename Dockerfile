FROM icepack/firedrake-python3.8:0.5.5

MAINTAINER shapero.daniel@gmail.com

RUN sudo apt-get update && \
    DEBIAN_FRONTEND="noninteractive" sudo apt-get install -yq \
    ffmpeg \
    graphviz \
    libgraphviz-dev

RUN mkdir actions-runner && \
    cd actions-runner && \
    curl -o actions-runner-linux-x64-2.285.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.285.1/actions-runner-linux-x64-2.285.1.tar.gz && \
    echo "5fd98e1009ed13783d17cc73f13ea9a55f21b45ced915ed610d00668b165d3b2  actions-runner-linux-x64-2.285.1.tar.gz" | shasum -a 256 -c && \
    tar xzf ./actions-runner-linux-x64-2.285.1.tar.gz && \
    ./config.sh --url https://github.com/danshapero/danshapero.github.io --token ${token} --unattended

ENTRYPOINT actions-runner/run.sh

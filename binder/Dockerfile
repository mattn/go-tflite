# mybinder.org launcher for the gophernotes image: reuse the prebuilt
# container and just add the unprivileged user that Binder runs as.
FROM ghcr.io/mattn/go-tflite/gophernotes:latest

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV HOME=/home/${NB_USER}

RUN useradd -m -u ${NB_UID} -s /bin/bash ${NB_USER} \
 && cp /usr/local/share/gophernotes-examples/* ${HOME}/ \
 && mkdir -p ${HOME}/.cache \
 && cp -r /root/.cache/go-build ${HOME}/.cache/go-build \
 && chown -R ${NB_UID} ${HOME} /go

ENTRYPOINT []
WORKDIR ${HOME}
USER ${NB_USER}

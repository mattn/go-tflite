# Jupyter + gophernotes with go-tflite ready to import.
#
#   docker build -t ghcr.io/mattn/gophernotes-tflite .
#   docker run --rm -p 8888:8888 ghcr.io/mattn/gophernotes-tflite
#
# The TensorFlow Lite C libraries come from the go-tflite buildkit that CI
# attaches to releases, so no bazel build is involved.

ARG GO_VERSION=1.25

FROM golang:${GO_VERSION}-trixie AS build

# The gomacro pinned by gophernotes is too old for current Go, so bump it
# to the latest before building.
ARG GOPHERNOTES_COMMIT=55142043d19696ba037e3e93f9ec6c7f8436e82d
RUN mkdir -p /src/gophernotes && cd /src/gophernotes \
 && git init -q \
 && git fetch -q --depth 1 https://github.com/gopherdata/gophernotes ${GOPHERNOTES_COMMIT} \
 && git checkout -q FETCH_HEAD \
 && go get github.com/cosmos72/gomacro@master \
 && go mod tidy \
 && go install -ldflags "-s -w" .

# The runtime image still needs the Go toolchain and gcc: gophernotes
# compiles imported packages (go-tflite included, via cgo) on the fly.
FROM debian:trixie-slim

ARG GO_TFLITE_VERSION=v1.0.7
ARG BUILDKIT_VERSION=v1.0.7

COPY --from=build /usr/local/go /usr/local/go
ENV GOPATH=/go
ENV PATH=/opt/jupyter/bin:/go/bin:/usr/local/go/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates gcc libc6-dev python3 python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/jupyter \
 && /opt/jupyter/bin/pip install --no-cache-dir notebook \
 && find /opt/jupyter -name __pycache__ -prune -exec rm -rf {} +

# TensorFlow Lite C API + XNNPACK delegate libraries and headers.
ADD https://github.com/mattn/go-tflite/releases/download/${BUILDKIT_VERSION}/go-tflite-buildkit-${BUILDKIT_VERSION}.tar.gz /tmp/buildkit.tar.gz
RUN tar -C /usr/local -xzf /tmp/buildkit.tar.gz && rm /tmp/buildkit.tar.gz && ldconfig

# gophernotes kernel.
COPY --from=build /go/bin/gophernotes /usr/local/bin/gophernotes
COPY --from=build /src/gophernotes/kernel /usr/local/share/jupyter/kernels/go

# Pre-warm the module and build caches so importing go-tflite in a notebook
# does not have to download or compile anything.
RUN mkdir -p /tmp/warm && cd /tmp/warm \
 && go mod init warm \
 && go get github.com/mattn/go-tflite@${GO_TFLITE_VERSION} \
 && printf 'package main\nimport (\n"github.com/mattn/go-tflite"\n"github.com/mattn/go-tflite/delegates/xnnpack"\n)\nfunc main() {\n_ = tflite.NewInterpreterOptions\n_ = xnnpack.New\n}\n' > main.go \
 && go build -o /dev/null . \
 && cd / && rm -rf /tmp/warm

WORKDIR /notebooks
COPY go-tflite.ipynb .
RUN cp "$(go env GOMODCACHE)/github.com/mattn/go-tflite@${GO_TFLITE_VERSION}/testdata/xor_model.tflite" .

EXPOSE 8888
CMD ["jupyter", "notebook", "--no-browser", "--allow-root", "--ip=0.0.0.0"]

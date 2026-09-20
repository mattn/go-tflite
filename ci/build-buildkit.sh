#!/bin/bash
# Build the go-tflite buildkit tarball.
#
# The buildkit is a tarball that provides everything needed for `go build` of
# go-tflite to succeed:
#
#   include/tensorflow/lite/...      TensorFlow Lite headers (only the ones
#                                    actually included, computed via gcc -M)
#   lib/libtensorflowlite_c.{so,dylib}
#                                    TensorFlow Lite C API shared library
#   lib/libtensorflowlite-delegate_xnnpack.{so,dylib}
#                                    XNNPACK delegate shared library
#   lib/libXNNPACK.{so,dylib}        XNNPACK itself
#
# On Linux and macOS the libraries are built with bazel and extracting the
# tarball into /usr/local is enough. On Windows they are built with cmake,
# either with MSVC (tensorflowlite_c.dll + tensorflowlite_c.lib) or with
# MinGW-w64 (libtensorflowlite_c.dll + libtensorflowlite_c.dll.a), and XNNPACK
# is compiled in; programs using delegates/xnnpack must then be built with
# `-tags xnnpack_builtin`.
#
# Run from the go-tflite repository root (Git Bash or an MSYS2 MINGW64 shell
# on Windows).
#
# Environment variables:
#   TENSORFLOW_VERSION  git tag/branch of tensorflow to build (default: v2.17.1)
#   TENSORFLOW_SRC      where to clone/find the tensorflow source
#   OUT_DIR             where to place the resulting tarball (default: ./dist)
#   BUILDKIT_SUFFIX     suffix of the tarball name, typically a go-tflite
#                       release tag (default: today's date as YYYYMMDD)
#   TOOLCHAIN           Windows only: msvc (default) or mingw
#   BAZEL_OUTPUT_USER_ROOT
#                       if set, passed to bazel as --output_user_root so that
#                       CI can cache it
set -euo pipefail

TENSORFLOW_VERSION=${TENSORFLOW_VERSION:-v2.17.1}
TENSORFLOW_SRC=${TENSORFLOW_SRC:-$HOME/tensorflow_src}
OUT_DIR=${OUT_DIR:-$PWD/dist}
GO_TFLITE_ROOT=$PWD

case "$(uname -s)" in
  Linux)  OS=linux ;;
  Darwin) OS=darwin ;;
  MINGW*|MSYS*|CYGWIN*) OS=windows ;;
  *) echo "unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64|amd64)  ARCH=amd64 ;;
  aarch64|arm64) ARCH=arm64 ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac
TOOLCHAIN=${TOOLCHAIN:-msvc}

if [ "$OS" = windows ]; then
  # Git Bash: normalize Windows-style paths coming from the environment.
  TENSORFLOW_SRC=$(cygpath -u "$TENSORFLOW_SRC")
  OUT_DIR=$(cygpath -u "$OUT_DIR")
fi
BAZEL_OUTPUT_USER_ROOT=${BAZEL_OUTPUT_USER_ROOT:-}
BAZEL_OUTPUT_USER_ROOT=${BAZEL_OUTPUT_USER_ROOT/#\~/$HOME}

if [ ! -d "$TENSORFLOW_SRC" ]; then
  git clone --depth 1 --branch "$TENSORFLOW_VERSION" \
    https://github.com/tensorflow/tensorflow "$TENSORFLOW_SRC"
fi

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/include" "$STAGE/lib"

cd "$TENSORFLOW_SRC"

build_bazel() {
  # Newer libc++ (Xcode 16.3+) rejects taking the address of std::abs<T>;
  # upstream replaced these with lambdas after 2.19, so do the same here.
  sed -i.bak \
    -e 's/std::abs<float>, type)/[](float f) { return std::abs(f); }, type)/' \
    -e 's/std::abs<int32_t>, type)/[](int32_t i) { return std::abs(i); }, type)/' \
    tensorflow/lite/kernels/elementwise.cc && rm -f tensorflow/lite/kernels/elementwise.cc.bak

  # Non-interactive configure: CPU only, no Android/iOS, default toolchain.
  export PYTHON_BIN_PATH=${PYTHON_BIN_PATH:-$(command -v python3)}
  export TF_NEED_CUDA=0
  export TF_NEED_ROCM=0
  export TF_NEED_CLANG=${TF_NEED_CLANG:-0}
  export TF_SET_ANDROID_WORKSPACE=0
  export TF_CONFIGURE_IOS=0
  export CC_OPT_FLAGS=${CC_OPT_FLAGS:--O2}
  python3 configure.py

  if [ "$OS" = darwin ]; then
    EXT=dylib
    # bazel 6's Apple toolchain wrapper (wrapped_clang) is linked without
    # LC_UUID and macOS 15.4+ refuses to run it; the plain Unix toolchain
    # drives clang directly and is enough for TensorFlow Lite.
    export BAZEL_USE_CPP_ONLY_TOOLCHAIN=1
    # Bake in the install location so binaries find the libraries without
    # DYLD_LIBRARY_PATH once the buildkit is extracted into /usr/local.
    SONAME_FLAG=-Wl,-install_name,/usr/local/lib/
  else
    EXT=so
    SONAME_FLAG=-Wl,-soname,
  fi

  # Upstream has no shared-library targets for the XNNPACK delegate, so append
  # them to the BUILD file (guarded so a re-run on the same checkout is a no-op).
  if ! grep -q 'libtensorflowlite-delegate_xnnpack' tensorflow/lite/delegates/xnnpack/BUILD; then
    cat >> tensorflow/lite/delegates/xnnpack/BUILD <<EOB

cc_binary(
    name = "libtensorflowlite-delegate_xnnpack.$EXT",
    linkopts = ["${SONAME_FLAG}libtensorflowlite-delegate_xnnpack.$EXT"],
    linkshared = True,
    deps = [":xnnpack_delegate"],
)

cc_binary(
    name = "libXNNPACK.$EXT",
    linkopts = ["${SONAME_FLAG}libXNNPACK.$EXT"],
    linkshared = True,
    deps = ["@XNNPACK//:XNNPACK"],
)
EOB
  fi

  local startup=()
  if [ -n "$BAZEL_OUTPUT_USER_ROOT" ]; then
    startup=(--output_user_root="$BAZEL_OUTPUT_USER_ROOT")
  fi
  bazel "${startup[@]}" build -c opt \
    //tensorflow/lite/c:tensorflowlite_c \
    "//tensorflow/lite/delegates/xnnpack:libtensorflowlite-delegate_xnnpack.$EXT" \
    "//tensorflow/lite/delegates/xnnpack:libXNNPACK.$EXT"

  install -m 755 \
    "bazel-bin/tensorflow/lite/c/libtensorflowlite_c.$EXT" \
    "bazel-bin/tensorflow/lite/delegates/xnnpack/libtensorflowlite-delegate_xnnpack.$EXT" \
    "bazel-bin/tensorflow/lite/delegates/xnnpack/libXNNPACK.$EXT" \
    "$STAGE/lib/"
  if [ "$OS" = darwin ]; then
    install_name_tool -id /usr/local/lib/libtensorflowlite_c.dylib \
      "$STAGE/lib/libtensorflowlite_c.dylib"
  fi
}

build_cmake() {
  # cmake is a native Windows program, so hand it Windows-style paths.
  local src build
  src=$(cygpath -m "$PWD/tensorflow/lite/c")
  build=$(cygpath -m "${TFLITE_BUILD:-$TENSORFLOW_SRC/../tflite_build}")
  # CMAKE_POLICY_VERSION_MINIMUM: some dependencies (FP16, psimd) still
  # declare cmake_minimum_required < 3.5, which CMake 4 rejects otherwise.
  # It has to be an environment variable to reach the nested cmake processes
  # that download them.
  export CMAKE_POLICY_VERSION_MINIMUM=3.5
  # MSVC only accepts the designated initializers used by TensorFlow Lite in
  # C++20 mode; upstream has since moved its cmake build to C++20 as well.
  sed -i.bak 's/set(CMAKE_CXX_STANDARD 17)/set(CMAKE_CXX_STANDARD 20)/' \
    tensorflow/lite/CMakeLists.txt tensorflow/lite/c/CMakeLists.txt
  rm -f tensorflow/lite/CMakeLists.txt.bak tensorflow/lite/c/CMakeLists.txt.bak
  local gen=()
  if [ "$TOOLCHAIN" = mingw ]; then
    gen=(-G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++)
  fi
  cmake "${gen[@]}" -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    -DTFLITE_ENABLE_XNNPACK=ON
  # cpuinfo relies on the max() macro from windows.h, which TensorFlow Lite's
  # cmake disables globally with NOMINMAX. MSVC's stdlib.h provides another
  # one, MinGW's does not.
  local cpuinfo_init=$build/cpuinfo/src/x86/windows/init.c
  if [ -f "$cpuinfo_init" ] && ! grep -q 'define max(' "$cpuinfo_init"; then
    sed -i '1i #define max(a, b) (((a) > (b)) ? (a) : (b))' "$cpuinfo_init"
  fi
  # gemmlowp adds MSVC-only compiler flags under a plain if(WIN32), which a
  # MinGW gcc does not understand.
  local gemmlowp_cmake=$build/gemmlowp/contrib/CMakeLists.txt
  if [ "$TOOLCHAIN" = mingw ] && [ -f "$gemmlowp_cmake" ]; then
    sed -i '/add_definitions(\/bigobj/d' "$gemmlowp_cmake"
  fi
  # The XNNPACK weight cache picks between an mmap and a read implementation
  # with _MSC_VER, so MinGW ends up on the POSIX branch and looks for
  # sys/mman.h. The Windows branch compiles fine with MinGW.
  if [ "$TOOLCHAIN" = mingw ]; then
    sed -i 's/defined(_MSC_VER)/defined(_WIN32)/g' \
      tensorflow/lite/delegates/xnnpack/weight_cache.cc
  fi
  cmake --build "$build" --config Release --target tensorflowlite_c -j
  # Multi-config generators (MSVC) put outputs under Release/.
  local dir=$build
  [ -d "$build/Release" ] && dir=$build/Release
  # MSVC: tensorflowlite_c.dll + tensorflowlite_c.lib
  # MinGW: libtensorflowlite_c.dll + libtensorflowlite_c.dll.a
  echo "cmake output in $dir:"
  ls -l "$dir" | grep -i tensorflowlite_c || true
  cp "$dir"/*tensorflowlite_c.dll "$STAGE/lib/"
  for f in "$dir"/*tensorflowlite_c.lib "$dir"/*tensorflowlite_c.dll.a; do
    [ -f "$f" ] && cp "$f" "$STAGE/lib/"
  done

  # cgo links with gcc even when the DLL itself was built by MSVC, and GNU ld
  # cannot always use an MSVC import library, so derive one from the DLL.
  if [ ! -f "$STAGE/lib/libtensorflowlite_c.dll.a" ] \
     && command -v objdump >/dev/null && command -v dlltool >/dev/null; then
    local dll=$STAGE/lib/tensorflowlite_c.dll
    { echo "LIBRARY tensorflowlite_c.dll"; echo EXPORTS; \
      objdump -p "$dll" | awk '
        /\[Ordinal\/Name Pointer\] Table/ { t = 1; next }
        t && /^[ \t]*\[ *[0-9]+\]/ { sub(/^[ \t]*\[ *[0-9]+\][ \t]*/, ""); print $1; next }
        t && NF == 0 { t = 0 }'; \
    } > "$STAGE/lib/tensorflowlite_c.def"
    local nexport
    nexport=$(($(wc -l < "$STAGE/lib/tensorflowlite_c.def") - 2))
    echo "exports found in tensorflowlite_c.dll: $nexport"
    if [ "$nexport" -lt 1 ]; then
      echo "could not read the export table of tensorflowlite_c.dll" >&2
      objdump -p "$dll" | head -40 >&2
      exit 1
    fi
    dlltool -d "$STAGE/lib/tensorflowlite_c.def" -D tensorflowlite_c.dll \
      -l "$STAGE/lib/libtensorflowlite_c.dll.a"
    rm -f "$STAGE/lib/tensorflowlite_c.def"
  fi
}


if [ "$OS" = windows ]; then
  build_cmake
else
  build_bazel
fi

# Collect only the headers go-tflite actually includes, plus their transitive
# includes, computed by the C preprocessor. Fall back to every header under
# tensorflow/lite when no gcc-compatible compiler is available.
grep -h '#include <tensorflow/' \
  "$GO_TFLITE_ROOT"/*.go.h \
  "$GO_TFLITE_ROOT"/delegates/xnnpack/*.go.h \
  | sed 's/<\(.*\)>/"\1"/' > "$STAGE/probe.c"
if command -v gcc >/dev/null 2>&1; then
  headers=$(gcc -M -I. "$STAGE/probe.c" | tr ' \\' '\n' | grep '^tensorflow/lite/' | sort -u)
else
  headers=$(find tensorflow/lite -name '*.h' | sort)
fi
echo "$headers" | while read -r f; do
  mkdir -p "$STAGE/include/$(dirname "$f")"
  cp "$f" "$STAGE/include/$f"
done
rm -f "$STAGE/probe.c"

mkdir -p "$OUT_DIR"
TARGET=$OS-$ARCH
[ "$OS" = windows ] && TARGET=$TARGET-$TOOLCHAIN
NAME=go-tflite-buildkit-${BUILDKIT_SUFFIX:-$(date +%Y%m%d)}-$TARGET.tar.gz
tar czf "$OUT_DIR/$NAME" -C "$STAGE" include lib
echo "created: $OUT_DIR/$NAME"
tar tzf "$OUT_DIR/$NAME" | grep -v '^include/' || true


.PHONY: build_library_darwin, build_library_linux
.PHONY: build_library_android_arm, build_library_android_arm64
.PHONY: build_library_ios_64, build_library_ios_simulator

IOS_ARCH=arm64
IPHONEOS_PLATFORM := $(shell xcrun --sdk iphoneos --show-sdk-platform-path)
IPHONEOS_SYSROOT := $(shell xcrun --sdk iphoneos --show-sdk-path)
MIN_SDK_VERSION=9.0
CC_FLAGS=$(shell xcrun --sdk iphoneos --find clang)

#IOS Simulator config
IPHONEOS_SYSROOT_SIMULATOR := $(shell xcrun --sdk iphonesimulator --show-sdk-path)
MIN_SDK_VERSION_SIMULATOR=12.0
IOS_ARCH_SIMULATOR=x86_64
CC_FLAGS_SIMULATOR=$(shell xcrun --sdk iphonesimulator --find clang)


build_library_darwin:
	CGO_ENABLED=1 GOOS=darwin GOARCH=amd64 go build -o $(CURDIR)/gen/darwin_amd64/libtextclassification.so -buildmode=c-shared .

build_library_linux:
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -o $(CURDIR)/gen/linux_amd64/libtextclassification.so -buildmode=c-shared .

build_library_android_arm:
	CGO_ENABLED=1 GOOS=android GOARCH=arm GOARM=7 CC=arm-linux-androideabi-gcc CXX=false \
	go build -o $(CURDIR)/gen/armeabi-v7a/libtextclassification.so -buildmode=c-shared .

build_library_android_arm64:
	CGO_LDFLAGS+="-fuse-ld=gold" \
	CGO_ENABLED=1 GOOS=android GOARCH=arm64 CC=aarch64-linux-android21-clang CXX=aarch64-linux-android21-clang++ \
	go build -o $(CURDIR)/gen/arm64-v8a/libtextclassification.so -buildmode=c-shared .

build_library_ios_64:
	CGO_CFLAGS+="-miphoneos-version-min=$(MIN_SDK_VERSION) \
		-isysroot ${IPHONEOS_SYSROOT} \
		-arch $(IOS_ARCH) \
		-O3" \
	CGO_LDFLAGS+="-miphoneos-version-min=$(MIN_SDK_VERSION) \
		-isysroot ${IPHONEOS_SYSROOT} \
		-arch $(IOS_ARCH) \
		-lc++" \
	CGO_ENABLED=1 \
	GOOS=darwin \
	GOARCH=$(IOS_ARCH) \
	CC=$(CC_FLAGS) \
	CXX=clang++ \
	go build -ldflags -w -v -installsuffix goi -o $(CURDIR)/gen/ios_$(IOS_ARCH)/libtextclassification.a -buildmode=c-archive .

build_library_ios_simulator:
	CGO_CFLAGS+="-isysroot ${IPHONEOS_SYSROOT_SIMULATOR} \
		-arch $(IOS_ARCH_SIMULATOR)" \
	CGO_LDFLAGS+="-mios-simulator-version-min=$(MIN_SDK_VERSION_SIMULATOR) \
		-isysroot ${IPHONEOS_SYSROOT_SIMULATOR} \
		-arch $(IOS_ARCH_SIMULATOR) \
		-lc++" \
	CGO_ENABLED=1 \
	GOOS=darwin \
	CC=cc \
	CXX=c++ 
	go build -ldflags -w -v -installsuffix opsim -o $(CURDIR)/gen/ios_$(IOS_ARCH_SIMULATOR)/libtextclassification.a -buildmode=c-archive .
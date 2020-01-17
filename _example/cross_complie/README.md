# Cross Compilation Example for Android, IOS etc...

By using  ```go -buildmode```  c-shared(Android, Linux) and c-archive(IOS) we can generate C library and the header of the Classifier.
The exported C methods ```Build()``` , ```Classify(void* p0, char* p1)``` and ```Close(void* p0)``` can be invoked from other platforms.

### Cross Compilation Setup

- Android : Setup the standalone toolchain using android ndk as explained here https://jasonplayne.com/programming-2/how-to-cross-compile-golang-for-android
- IOS : Install xcode and toolchain will be available for compilation

### Cross Compile
- Darwin amd64(x86_64) : ```make build_library_darwin```
- Linux amd64(x86_64) : ```make build_library_linux```
- Android armeabi-v7a : ```make build_library_android_arm```
- Android arm64-v8a : ```make build_library_android_arm64```
- IOS arm64 : ```make build_library_ios_64```
- IOS Simulator amd64(x86_64) : ```make build_library_ios_simulator```


### Tested on macOS 10.15.2 and Tensorflow Lite C libraries generated from tensorflow master on 17th January 2020.


### Note
Once accepted and merged you can remove ```replace github.com/mattn/go-tflite => ../../../go-tflite``` from go.mod and update the ```github.com/mattn/go-tflite``` version.
set -eu

SOURCE_DIR="$PWD"

# clean:
#find -type d -name build -print -exec rm --recursive {} +

build() {
	ARCH="$1"
	ABI="$2"

	mkdir --parents "build/$ABI"
	pushd "build/$ABI"
	ANDROID_SDK="$ANDROID_SDK" \
	ANDROID_NDK="$ANDROID_NDK" \
	cmake "$SOURCE_DIR" \
		-DANDROID_WRAPPER=ON \
		-DCMAKE_TOOLCHAIN_FILE="$SOURCE_DIR/android.toolchain.cmake" \
		-DANDROID_ABI="$ABI" \
		-DOpenCV_DIR="$OPENCV_ANDROID_SDK/sdk/native/jni" \
		-DJAVA_INCLUDE_PATH2="$ANDROID_NDK/platforms/android-9/arch-$ARCH/usr/include" \
		-DJAVA_AWT_INCLUDE_PATH="$ANDROID_NDK/platforms/android-9/arch-$ARCH/usr/include" \
		-DOpenCV_ANDROID_SDK="$OPENCV_ANDROID_SDK/sdk" \
		-DANDROID_NATIVE_API_LEVEL=android-9
	make -j
	popd
}

build arm armeabi-v7a
build arm64 arm64-v8a

pushd NDKTest
./gradlew build
popd

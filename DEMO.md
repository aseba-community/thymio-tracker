* Checkout this branch
`git checkout demo`
* Update submodules
`git submodule update --init`
* Build the APK
`OPENCV_ANDROID_SDK=... ANDROID_SDK=... ANDROID_NDK=... ./android.sh`
* Put config files on the device
`adb push data/Config.xml data/calibration.xml data/GHscale_Arth_Perspective.xml data/modelSurfaces.xml /sdcard/ThymioTracker/`
* Install the APK
`adb install ./NDKTest/app/build/outputs/apk/app-release-unsigned.apk`
* Run the calibration to update the calibration.xml file
* Turn on the Thymio
* Plug the dongle
* Run the tracker

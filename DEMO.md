* Checkout this branch
`git checkout demo`
* Update submodules
`git submodule update --init`
* Remove uin64 declaration and replace usages with uint64_t in aseba
https://github.com/aseba-community/aseba/issues/604
* Build the APK
`OPENCV_ANDROID_SDK=... ANDROID_SDK=... ANDROID_NDK=... ./android.sh`
* Put config files on the device
`adb push data/Config.xml data/calibration.xml data/GHscale_Arth_Perspective.xml data/modelSurfaces.xml /sdcard/ThymioTracker/`
* Start the Aseba switch
`asebaswitch -v ser:name=Thymio-II`
* Configure the connection between the device and the switch
Either `adb reverse tcp:33333 tcp:33333` or change the `dashelTarget` in Config.xml
* Import project NDKTest in android studio
* Click play
* Run the calibration to update the calibration.xml file

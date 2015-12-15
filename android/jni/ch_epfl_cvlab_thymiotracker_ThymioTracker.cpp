
#include "ch_epfl_cvlab_thymiotracker_ThymioTracker.h"

#include "ThymioTracker.h"

#ifdef __cplusplus
extern "C" {
#endif

using namespace thymio_tracker;

static void GetJStringContent(JNIEnv *AEnv, jstring AStr, std::string &ARes)
{
    if (!AStr)
    {
        ARes.clear();
        return;
    }

    const char *s = AEnv->GetStringUTFChars(AStr,NULL);
    ARes=s;
    AEnv->ReleaseStringUTFChars(AStr,s);
}

/*
 * Class:     ch_epfl_cvlab_thymiotracker_ThymioTracker
 * Method:    createNativeInstance
 * Signature: (Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ch_epfl_cvlab_thymiotracker_ThymioTracker_createNativeInstance
  (JNIEnv * env, jobject, jstring _calibrationFile, jstring _geomHashingFile)
{
    std::string calibrationFile;
    std::string geomHashingFile;
    
    GetJStringContent(env, _calibrationFile, calibrationFile);
    GetJStringContent(env, _geomHashingFile, geomHashingFile);
    
    return reinterpret_cast<long>(new ThymioTracker(calibrationFile, geomHashingFile));
}

/*
 * Class:     ch_epfl_cvlab_thymiotracker_ThymioTracker
 * Method:    destroyNativeInstance
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ch_epfl_cvlab_thymiotracker_ThymioTracker_destroyNativeInstance
  (JNIEnv *, jobject, jlong ptr_ttracker)
{
    delete reinterpret_cast<ThymioTracker*>(ptr_ttracker);
}

/*
 * Class:     ch_epfl_cvlab_thymiotracker_ThymioTracker
 * Method:    n_update
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ch_epfl_cvlab_thymiotracker_ThymioTracker_n_1update__JJ
  (JNIEnv *, jobject, jlong ptr_ttracker, jlong ptr_input)
{
    ThymioTracker* ttracker = reinterpret_cast<ThymioTracker*>(ptr_ttracker);
    cv::Mat* input = reinterpret_cast<cv::Mat*>(ptr_input);
    return ttracker->update(*input);
}

/*
 * Class:     ch_epfl_cvlab_thymiotracker_ThymioTracker
 * Method:    n_update
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_ch_epfl_cvlab_thymiotracker_ThymioTracker_n_1update__JJJ
  (JNIEnv *, jobject, jlong ptr_ttracker, jlong ptr_input, jlong ptr_deviceOrientation)
{
    ThymioTracker* ttracker = reinterpret_cast<ThymioTracker*>(ptr_ttracker);
    cv::Mat* input = reinterpret_cast<cv::Mat*>(ptr_input);
    cv::Mat* deviceOrientation = reinterpret_cast<cv::Mat*>(ptr_deviceOrientation);
    return ttracker->update(*input, deviceOrientation);
}

/*
 * Class:     ch_epfl_cvlab_thymiotracker_ThymioTracker
 * Method:    n_drawLastDetection
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ch_epfl_cvlab_thymiotracker_ThymioTracker_n_1drawLastDetection
  (JNIEnv *, jobject, jlong ptr_ttracker, jlong ptr_output)
{
    ThymioTracker* ttracker = reinterpret_cast<ThymioTracker*>(ptr_ttracker);
    cv::Mat* output = reinterpret_cast<cv::Mat*>(ptr_output);
    ttracker->drawLastDetection(output);
}

/*
 * Class:     ch_epfl_cvlab_thymiotracker_ThymioTracker
 * Method:    n_setScale
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_ch_epfl_cvlab_thymiotracker_ThymioTracker_n_1setScale
  (JNIEnv *, jobject, jlong ptr_ttracker, jdouble scale)
{
    ThymioTracker* ttracker = reinterpret_cast<ThymioTracker*>(ptr_ttracker);
    ttracker->setScale(scale);
}

#ifdef __cplusplus
}
#endif

#include "ch_epfl_cvlab_calibrator_Calibrator.h"

#include "Calibrator.h"

#ifdef __cplusplus
extern "C" {
#endif

using namespace thymio_tracker;

static std::string GetJStringContent(JNIEnv *AEnv, jstring AStr)
{
    if (!AStr)
        return std::string();
    
    const char *s = AEnv->GetStringUTFChars(AStr,NULL);
    std::string res(s);
    AEnv->ReleaseStringUTFChars(AStr,s);
    
    return res;
}

/*
 * Class:     ch_epfl_cvlab_calibrator_Calibrator
 * Method:    createNativeInstance
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ch_epfl_cvlab_calibrator_Calibrator_createNativeInstance
  (JNIEnv * env, jobject, jstring _configFile)
{
    std::string configFile = GetJStringContent(env, _configFile);
    
    return reinterpret_cast<long>(new Calibrator(configFile));
}

/*
 * Class:     ch_epfl_cvlab_calibrator_Calibrator
 * Method:    destroyNativeInstance
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ch_epfl_cvlab_calibrator_Calibrator_destroyNativeInstance
  (JNIEnv *, jobject, jlong ptr_ttracker)
{
    delete reinterpret_cast<Calibrator*>(ptr_ttracker);
}

/*
 * Class:     ch_epfl_cvlab_calibrator_Calibrator
 * Method:    n_update
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ch_epfl_cvlab_calibrator_Calibrator_n_1update
  (JNIEnv *, jobject, jlong ptr_ttracker, jlong ptr_input)
{
    Calibrator* ttracker = reinterpret_cast<Calibrator*>(ptr_ttracker);
    cv::Mat* input = reinterpret_cast<cv::Mat*>(ptr_input);
    return ttracker->update(*input);
}
/*
 * Class:     ch_epfl_cvlab_calibrator_Calibrator
 * Method:    n_drawLastDetection
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ch_epfl_cvlab_calibrator_Calibrator_n_1drawLastDetection
  (JNIEnv *, jobject, jlong ptr_ttracker, jlong ptr_output)
{
    Calibrator* ttracker = reinterpret_cast<Calibrator*>(ptr_ttracker);
    cv::Mat* output = reinterpret_cast<cv::Mat*>(ptr_output);
    ttracker->drawState(output);
}

#ifdef __cplusplus
}
#endif

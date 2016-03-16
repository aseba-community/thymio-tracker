
#include "ch_epfl_cvlab_thymiotracker_ThymioTracker.h"

#include "ThymioTracker.h"

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
 * Class:     ch_epfl_cvlab_thymiotracker_ThymioTracker
 * Method:    createNativeInstance
 * Signature: (Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ch_epfl_cvlab_thymiotracker_ThymioTracker_createNativeInstance__Ljava_lang_String_2Ljava_lang_String_2
  (JNIEnv * env, jobject, jstring _calibrationFile, jstring _geomHashingFile)
{
    std::string calibrationFile = GetJStringContent(env, _calibrationFile);
    std::string geomHashingFile = GetJStringContent(env, _geomHashingFile);
    
    return reinterpret_cast<long>(new ThymioTracker(calibrationFile, geomHashingFile));
}

JNIEXPORT jlong JNICALL Java_ch_epfl_cvlab_thymiotracker_ThymioTracker_createNativeInstance__Ljava_lang_String_2Ljava_lang_String_2_3Ljava_lang_String_2
  (JNIEnv * env, jobject, jstring _calibrationFile, jstring _geomHashingFile, jobjectArray _markerFiles)
{
    std::string calibrationFile = GetJStringContent(env, _calibrationFile);
    std::string geomHashingFile = GetJStringContent(env, _geomHashingFile);
    std::vector<std::string> markerFiles;
    
    int stringCount = env->GetArrayLength(_markerFiles);
    for(int i = 0; i < stringCount; ++i)
    {
        jstring string = (jstring) env->GetObjectArrayElement(_markerFiles, i);
        
        std::string markerFile = GetJStringContent(env, string);
        markerFiles.push_back(markerFile);
    }
    
    return reinterpret_cast<long>(new ThymioTracker(calibrationFile, geomHashingFile, markerFiles));
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

#ifdef __cplusplus
}
#endif

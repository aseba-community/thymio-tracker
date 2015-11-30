
#include "ch_epfl_cvlab_arthymio_ARThymio.h"

#include "arthymio.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     ch_epfl_cvlab_arthymio_ARThymio
 * Method:    n_process
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_ch_epfl_cvlab_arthymio_ARThymio_n_1process__JJ
  (JNIEnv *, jclass, jlong ptr_input, jlong ptr_output)
{
    cv::Mat* input = reinterpret_cast<cv::Mat*>(ptr_input);
    cv::Mat* output = reinterpret_cast<cv::Mat*>(ptr_output);
    return process(*input, *output);
}

/*
 * Class:     ch_epfl_cvlab_arthymio_ARThymio
 * Method:    n_process
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_ch_epfl_cvlab_arthymio_ARThymio_n_1process__JJJ
  (JNIEnv *, jclass, jlong ptr_input, jlong ptr_output, jlong ptr_deviceOrientation)
{
    cv::Mat* input = reinterpret_cast<cv::Mat*>(ptr_input);
    cv::Mat* output = reinterpret_cast<cv::Mat*>(ptr_output);
    cv::Mat* deviceOrientation = reinterpret_cast<cv::Mat*>(ptr_deviceOrientation);
    return process(*input, *output, deviceOrientation);
}

/*
 * Class:     ch_epfl_cvlab_arthymio_ARThymio
 * Method:    n_get_rows
 * Signature: (J)I
 */
// JNIEXPORT jint JNICALL Java_ch_epfl_cvlab_arthymio_ARThymio_n_1get_1rows(JNIEnv *, jclass, jlong ptr_mat)
// {
//     cv::Mat* mat = reinterpret_cast<cv::Mat*>(ptr_mat);
//     return get_rows(*mat);
// }

#ifdef __cplusplus
}
#endif

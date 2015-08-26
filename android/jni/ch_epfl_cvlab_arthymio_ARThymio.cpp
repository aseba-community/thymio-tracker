
#include "ch_epfl_cvlab_arthymio_ARThymio.h"

#include "arthymio.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNICALL Java_ch_epfl_cvlab_arthymio_ARThymio_sum(JNIEnv *, jclass, jint a, jint b)
{
    return sum(a, b);
}

/*
 * Class:     ch_epfl_cvlab_arthymio_ARThymio
 * Method:    n_get_rows
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_ch_epfl_cvlab_arthymio_ARThymio_n_1get_1rows(JNIEnv *, jclass, jlong ptr_mat)
{
    cv::Mat* mat = reinterpret_cast<cv::Mat*>(ptr_mat);
    return get_rows(*mat);
}

#ifdef __cplusplus
}
#endif

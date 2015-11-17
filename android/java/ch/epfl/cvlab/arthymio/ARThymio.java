package ch.epfl.cvlab.arthymio;

import org.opencv.core.Mat;

public class ARThymio
{
    public static int process(Mat input, Mat output)
    {
        return n_process(input.nativeObj, output.nativeObj);
    }
    
    public static int process(Mat input, Mat output,
        Mat deviceOrientation)
    {
        return n_process(input.nativeObj, output.nativeObj,
                         deviceOrientation.nativeObj);
    }
    
    public static native int n_process(long input, long output);
    public static native int n_process(long input, long output,
                                       long deviceOrientation);
}

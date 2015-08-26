package ch.epfl.cvlab.arthymio;

import org.opencv.core.Mat;

public class ARThymio
{
    public static int get_rows(Mat mat) {
        return n_get_rows(mat.nativeObj);
    }
    
    public static native int sum(int a, int b);
    public static native int n_get_rows(long mat);
}

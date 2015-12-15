package ch.epfl.cvlab.thymiotracker;

import org.opencv.core.Mat;

public class ThymioTracker
{
    private long internalPtr;
    
    public ThymioTracker(String calibrationFile, String geomHashingFile)
    {
        this.internalPtr = createNativeInstance(calibrationFile, geomHashingFile);
    }
    
    public void finalize()
    {
        destroyNativeInstance(this.internalPtr);
    }
    
    public void update(Mat input)
    {
        n_update(this.internalPtr, input.nativeObj);
    }
    
    public void update(Mat input, Mat deviceOrientation)
    {
        n_update(this.internalPtr, input.nativeObj, deviceOrientation.nativeObj);
    }
    
    public void drawLastDetection(Mat output)
    {
        n_drawLastDetection(this.internalPtr, output.nativeObj);
    }
    
    public void setScale(double scale)
    {
        n_setScale(this.internalPtr, scale);
    }
    
    private native long createNativeInstance(String calibrationFile, String geomHashingFile);
    private native void destroyNativeInstance(long internalPtr);
    private native void n_update(long internalPtr, long input);
    private native void n_update(long internalPtr, long input, long deviceOrientation);
    private native void n_drawLastDetection(long internalPtr, long output);
    private native void n_setScale(long internalPtr, double scale);
}

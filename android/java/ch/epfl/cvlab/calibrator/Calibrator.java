package ch.epfl.cvlab.calibrator;

import org.opencv.core.Mat;

public class Calibrator
{
    private long internalPtr;
    
    public Calibrator(String calibrationFile)
    {
        this.internalPtr = createNativeInstance(calibrationFile);
    }
    
    public void finalize()
    {
        destroyNativeInstance(this.internalPtr);
    }
    
    public void update(Mat input)
    {
        n_update(this.internalPtr, input.nativeObj);
    }

    public void drawState(Mat output)
    {
        n_drawLastDetection(this.internalPtr, output.nativeObj);
    }

    
    private native long createNativeInstance(String configFile);
    private native void destroyNativeInstance(long internalPtr);
    private native void n_update(long internalPtr, long input);
    private native void n_drawLastDetection(long internalPtr, long output);
}

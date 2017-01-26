package ch.epfl.cvlab.ndktest;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.Uri;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;

// import ch.epfl.cvlab.arthymio.ARThymio;
import com.google.android.gms.appindexing.Action;
import com.google.android.gms.appindexing.AppIndex;
import com.google.android.gms.common.api.GoogleApiClient;

import ch.epfl.cvlab.thymiotracker.ThymioTracker;
import ch.epfl.cvlab.calibrator.Calibrator;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class MainActivity
        extends AppCompatActivity
        implements CvCameraViewListener, SensorEventListener {

    private static final String TAG = "NDKTest::MainActivity";

    private CameraBridgeViewBase mOpenCvCameraView;

    private ThymioTracker mThymioTracker;
    private Calibrator mCalibrator;

    private Mat mGrayImage = null;
    private SensorManager mSensorManager = null;
    private Sensor mRotationSensor = null;
    private Mat mRotationMatrix = new Mat(3, 3, CvType.CV_32FC1);

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    /* Now enable camera view to start receiving frames */
                    mOpenCvCameraView.enableView();
                    mGrayImage = new Mat();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };
    /**
     * ATTENTION: This was auto-generated to implement the App Indexing API.
     * See https://g.co/AppIndexing/AndroidStudio for more information.
     */
    private GoogleApiClient client;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // setContentView(R.layout.activity_main);

        final String extStorageDir = Environment.getExternalStorageDirectory().getPath();
        Log.d(TAG, "Storage Dir is " + extStorageDir);

        //mThymioTracker = new ThymioTracker(extStorageDir + "/ThymioTracker/");
        mCalibrator = new Calibrator(extStorageDir + "/ThymioTracker/newCalib.xml");

        mOpenCvCameraView = new JavaCameraView(this, -1);
        mOpenCvCameraView.setMaxFrameSize(800, 600);

        setContentView(mOpenCvCameraView);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mRotationSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);

        if (mRotationSensor == null) {
            Log.w(TAG, "No rotation tensor found!");
        }
        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        client = new GoogleApiClient.Builder(this).addApi(AppIndex.API).build();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        mSensorManager.registerListener(this, mRotationSensor, SensorManager.SENSOR_DELAY_NORMAL);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

        mSensorManager.unregisterListener(this);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        super.onCreateOptionsMenu(menu);
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    // CvCameraViewListener methods
    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(Mat inputFrame) {
        // Log.d(TAG, "Rows: " + ARThymio.get_rows(inputFrame));
        Imgproc.cvtColor(inputFrame, mGrayImage, Imgproc.COLOR_BGR2GRAY);
        // Log.d(TAG, "Input image size: " + inputFrame.rows() + " x " + inputFrame.cols());

        // ARThymio.process(inputFrame, mGrayImage, mRotationMatrix);

        //mThymioTracker.update(mGrayImage);
        //mThymioTracker.drawLastDetection(inputFrame, mRotationMatrix);

        mCalibrator.update(mGrayImage);
        mCalibrator.drawState(inputFrame);


//        if (mGrayImage.size() != inputFrame.size())
//            Imgproc.resize(mGrayImage, mGrayImage, inputFrame.size());

        return inputFrame;
    }

    static {
        System.loadLibrary("thymiotracker");
        System.loadLibrary("opencv_java3");
    }

    // SensorEventListener methods
    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR) {
            float[] rotationMatrix = new float[9];
            SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values);
            //SensorManager.remapCoordinateSystem(rotationMatrix,
            //        SensorManager.AXIS_X, SensorManager.AXIS_Y,
            //        rotationMatrix);

            mRotationMatrix.put(0, 0, rotationMatrix);
            Log.d(TAG, "Rotation is " + mRotationMatrix.dump() + ".");

            float[] orientationValuesV = new float[3];
            SensorManager.getOrientation(rotationMatrix, orientationValuesV);

            Log.d(TAG, "Orientation is " + orientationValuesV[0] + "," + orientationValuesV[1] + "," + orientationValuesV[2]);

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override
    public void onStart() {
        super.onStart();

        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        client.connect();
        Action viewAction = Action.newAction(
                Action.TYPE_VIEW, // TODO: choose an action type.
                "Main Page", // TODO: Define a title for the content shown.
                // TODO: If you have web page content that matches this app activity's content,
                // make sure this auto-generated web page URL is correct.
                // Otherwise, set the URL to null.
                Uri.parse("http://host/path"),
                // TODO: Make sure this auto-generated app deep link URI is correct.
                Uri.parse("android-app://ch.epfl.cvlab.ndktest/http/host/path")
        );
        AppIndex.AppIndexApi.start(client, viewAction);
    }

    @Override
    public void onStop() {
        super.onStop();

        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        Action viewAction = Action.newAction(
                Action.TYPE_VIEW, // TODO: choose an action type.
                "Main Page", // TODO: Define a title for the content shown.
                // TODO: If you have web page content that matches this app activity's content,
                // make sure this auto-generated web page URL is correct.
                // Otherwise, set the URL to null.
                Uri.parse("http://host/path"),
                // TODO: Make sure this auto-generated app deep link URI is correct.
                Uri.parse("android-app://ch.epfl.cvlab.ndktest/http/host/path")
        );
        AppIndex.AppIndexApi.end(client, viewAction);
        client.disconnect();
    }
}

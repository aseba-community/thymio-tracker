package ch.epfl.cvlab.ndktest;

import android.os.Bundle;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.File;

import ch.epfl.cvlab.calibrator.Calibrator;

public class CalibratorActivity
		extends MainActivity {

	private Calibrator mCalibrator;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		File calibratorFile = new File(trackerDirectory, "calibration.xml");
		mCalibrator = new Calibrator(calibratorFile.getPath());
	}

	public Mat onCameraFrame(Mat inputFrame) {
		Imgproc.cvtColor(inputFrame, mGrayImage, Imgproc.COLOR_BGR2GRAY);

		mCalibrator.update(mGrayImage);
		mCalibrator.drawState(inputFrame);

		return inputFrame;
	}
}

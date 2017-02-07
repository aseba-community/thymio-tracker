package ch.epfl.cvlab.ndktest;

import android.os.Bundle;

import org.mobsya.thymiovpl.DashelSerial;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.File;

import ch.epfl.cvlab.thymiotracker.ThymioTracker;

public class TrackerActivity
		extends MainActivity {

	private ThymioTracker mThymioTracker;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		DashelSerial.activity = this;
		mThymioTracker = new ThymioTracker(trackerDirectory.getPath() + "/");
	}

	public Mat onCameraFrame(Mat inputFrame) {
		Imgproc.cvtColor(inputFrame, mGrayImage, Imgproc.COLOR_BGR2GRAY);

		mThymioTracker.update(mGrayImage);
		mThymioTracker.drawLastDetection(inputFrame, mRotationMatrix);

		return inputFrame;
	}

	@Override
	protected void onDestroy() {
		super.onDestroy();
		mThymioTracker.finalize();
		DashelSerial.activity = null;
	}
}

package tr.edu.ozu.handwrittenmathexpressionsolver;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;


/**
 * This is the main class which does all the pre-processing and segmentation.
 * Heavily used Computer Vision / Image Processing concepts like Edge Detection,
 * Contour Finding, Filtering etc.
 * Also calls the detector class to perform recognition of the segmented digits
 */

public class ImageProcessor {

    private static final String TAG = "ImageProcessor";
    ArrayList<RoiObject> mRoiImages = new ArrayList<>(50);
    String mResultText;
    String mProbText;
    double otsu;
    public static final String[] symbols = {"+", "-", "x", "/"};

    // Amplify the region of interest by making the number bright and
    // background black. This removes noise due to shadows / insufficient lighting.
    // Used otsu thresholding for best result.
    public Mat preProcessImage(Bitmap image) {
        Size sz = new Size(640, 480);
        ArrayList<Rect> rects;
        Rect rect;

        int top,bottom,left,right;
        Bitmap.Config conf = Bitmap.Config.ARGB_8888; // see other conf types
        Mat origImageMatrix = new Mat(image.getWidth(), image.getHeight(), CvType.CV_8UC3);
        Mat tempImageMat = new Mat(image.getWidth(),image.getHeight(),CvType.CV_8UC1,new Scalar(0));
        Utils.bitmapToMat(image,origImageMatrix);

        Mat imgToProcess = new Mat (image.getWidth(), image.getHeight(), CvType.CV_8UC1);
        Mat imgToProcessCanny = new Mat (image.getWidth(), image.getHeight(), CvType.CV_8UC1);
        Mat imgToProcessOtsu = new Mat (image.getWidth(), image.getHeight(), CvType.CV_8UC1);
        Utils.bitmapToMat(image,imgToProcess);

        //Resize image to manageable size
        Imgproc.resize(imgToProcess, imgToProcess, sz,0,0,Imgproc.INTER_NEAREST);
        Imgproc.resize(origImageMatrix, origImageMatrix, sz,0,0,Imgproc.INTER_NEAREST);
        Imgproc.resize(tempImageMat, tempImageMat, sz,0,0,Imgproc.INTER_NEAREST);
        Imgproc.cvtColor(imgToProcess, imgToProcess, Imgproc.COLOR_BGR2GRAY);


        //Remove noise using Gaussian filter
        http://docs.opencv.org/2.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html
        Imgproc.GaussianBlur(imgToProcess,imgToProcess,new Size(3,3),0);


        Mat imgGrayInv = new Mat(sz,CvType.CV_8UC1,new Scalar(255.0));

        //Invert the brightness - make lighter pixel dark and vice versa.
        // https://en.wikipedia.org/wiki/Canny_edge_detector
        // Otsu's method [11] can be used on the non-maximum suppressed gradient magnitude image to generate the high threshold.
        // The low threshold is typically set to 1/2 of the high threshold in this case
        Core.subtract(imgGrayInv,imgToProcess,imgGrayInv);
        otsu = Imgproc.threshold(imgToProcess, imgToProcessOtsu,127,255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        Log.d("ANIL", "otsu = " + otsu);
        Imgproc.Canny(imgToProcess,imgToProcessCanny,otsu/2,otsu,3,false);
        rects = this.boundingBox(imgToProcessCanny);

        Log.d(TAG,"Length of rects : " + rects.size());

        if (rects.size() != 0) {
            rect = rects.get(0);
            top = rect.y;
            bottom = rect.y + rect.height;
            left = rect.x;
            right = rect.x + rect.height;
            for (int i = 1; i < rects.size(); i++) {
                rect = rects.get(i);
                if (rect.y < top) {
                    top = rect.y;
                }
                if (rect.y + rect.height > bottom) {
                    bottom = rect.y + rect.height;
                }
                if (rect.x < left) {
                    left = rect.x;
                }
                if (rect.x + rect.width > right) {
                    right = rect.x + rect.width;
                }
            }

            // Using the algorithm in the following paper to get the region of interest
            //http://web.stanford.edu/class/cs231m/projects/final-report-yang-pu.pdf

            Mat aux = tempImageMat.colRange(left, right).rowRange(top, bottom);
            MatOfDouble matMean = new MatOfDouble();
            MatOfDouble matStd = new MatOfDouble();
            Mat roiImage = imgGrayInv.submat(top, bottom, left, right).clone();
            Core.meanStdDev(roiImage, matMean, matStd);
            Imgproc.threshold(roiImage, roiImage,127,255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
            roiImage.copyTo(aux);
        }

        sz = new Size(image.getWidth(),image.getHeight());
        Imgproc.resize(tempImageMat,tempImageMat,sz);
        return tempImageMat;
    }

    public ArrayList<Rect> boundingBox(Mat imgToProcess) {
        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        ArrayList<Rect> rects = new ArrayList<>(50);
        Mat hierarchy = new Mat();
        Bitmap.Config conf = Bitmap.Config.ARGB_8888; // see other conf type
        Imgproc.findContours(imgToProcess, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));

        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {

            double contourArea = Imgproc.contourArea(contours.get(contourIdx));
            if ((contourArea < 5000) && (contourArea > 800000)) {
                Log.d(TAG,"fail boundingbox ContourArea = " + contourArea);
                continue;
            }
            Log.d(TAG,"pass bounding box ContourArea = " + contourArea);
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            MatOfPoint2f contour2f = new MatOfPoint2f(contours.get(contourIdx).toArray());
            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f, true) * 0.02;
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

            //Convert back to MatOfPoint
            MatOfPoint points = new MatOfPoint(approxCurve.toArray());

            // Get bounding rect of contour
            Rect rect = Imgproc.boundingRect(points);
            Log.d(TAG, "Rect Height :" + rect.height);

			if (rect.height > 5) {
            //if (rect.height > 50) {
                Log.d(TAG, "Adding Rect Height :" + rect.height);
                rects.add(rect);
            }


        }
        filterRectangles(rects);
        return rects;
    }



    public Mat segmentAndRecognize(Mat imgToProcess,Bitmap origImage,Classifier mClassifier) {
        StringBuilder ResultExpression = new StringBuilder("");
        StringBuilder ProbDigits  = new StringBuilder("");
        mRoiImages.clear();
        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Mat origImageMatrix = new Mat(origImage.getWidth(), origImage.getHeight(), CvType.CV_8UC3);
        Bitmap.Config conf = Bitmap.Config.ARGB_8888; // see other conf types
        Mat roiImage;
        Utils.bitmapToMat(origImage, origImageMatrix);
        Imgproc.findContours(imgToProcess, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));
        Result result;
        int label;

        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {

            double contourArea = Imgproc.contourArea(contours.get(contourIdx));

            // Filter out unwanted contour areas
            Log.d(TAG,"check ContourArea = " + contourArea + " counter num: " + contourIdx );

            if ((contourArea < 5000.0) && (contourArea > 800000.0)) {
                Log.v(TAG,"fail ContourArea = " + contourArea + " counter num: " + contourIdx );
                continue;
            }
            Log.v(TAG,"pass ContourArea = " + contourArea + " counter num: " + contourIdx );

            MatOfPoint2f approxCurve = new MatOfPoint2f();
            MatOfPoint2f contour2f = new MatOfPoint2f(contours.get(contourIdx).toArray());

            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f, true) * 0.02;
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

            //Convert back to MatOfPoint
            MatOfPoint points = new MatOfPoint(approxCurve.toArray());

            // Get bounding rectangle of contour
            Rect rect = Imgproc.boundingRect(points);
            Imgproc.rectangle(origImageMatrix, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0, 255), 3);
            if ((rect.y + rect.height > origImageMatrix.rows()) || (rect.x + rect.width > origImageMatrix.cols())) {
                Log.v(TAG, "Eliminate due to discrepancy between origimage and processed one !!!!! ");
                continue;
            }

            MatOfDouble matMean = new MatOfDouble();
            MatOfDouble matStd = new MatOfDouble();
            Double mean;

            roiImage = imgToProcess.submat(rect.y ,rect.y + rect.height , rect.x,rect.x + rect.width );
            int xCord = rect.x;
            Core.copyMakeBorder(roiImage, roiImage, 100, 100, 100, 100, Core.BORDER_ISOLATED);

            // Resize the roi image to 28x28 as our training dataset.
            Size sz = new Size(28, 28);
            Imgproc.resize(roiImage, roiImage, sz);


            Core.meanStdDev(roiImage, matMean, matStd);
            mean = matMean.toArray()[0];
            Imgproc.threshold(roiImage, roiImage, 0, 255, Imgproc.THRESH_BINARY_INV);
            Bitmap tempImage = Bitmap.createBitmap(roiImage.cols(), roiImage.rows(), conf);
            Utils.matToBitmap(roiImage, tempImage);
            RoiObject roiObject = new RoiObject(xCord,tempImage);
            mRoiImages.add(roiObject);
        }

        // To read the digits from left to right - sort as per the X coordinates.
        Collections.sort(mRoiImages);

        //Set the max number of digits to read to 9 (arbitrarily chosen)
        int max = (mRoiImages.size() > 20) ? 20 : mRoiImages.size();
        for (int i = 0; i < max; i++) {
            RoiObject roi = mRoiImages.get(i);

            result = mClassifier.classify(roi.bmp);
            label = result.getNumber();
            switch (label){
                case 10:
                    ResultExpression.append("+");
                    break;
                case 11:
                    ResultExpression.append("-");
                    break;
                case 12:
                    ResultExpression.append("x");
                    break;
                case 13:
                    ResultExpression.append("/");
                    break;
                case 14:
                    ResultExpression.append("(");
                    break;
                case 15:
                    ResultExpression.append(")");
                    break;
                default:
                    ResultExpression.append(label);
                    break;
            }
            Log.i(TAG, "digit =" + label);
            if(ProbDigits.length() != 0)
                ProbDigits.append(", ");
            ProbDigits.append(result.getProbability());
        }
        Log.i(TAG,"Numbers = :" + ResultExpression.toString());
        mResultText = divDetector(ResultExpression.toString());
        mProbText = ProbDigits.toString();
        return origImageMatrix;
    }

    public String getResultText() {
        return this.mResultText;
    }

    public String getProbText() {
        return this.mProbText;
    }

    public String divDetector(String expression) {
        int exp_index = 0;
        int next_index = 0;
        int prev_index = 0;
        for (String s : symbols) {

            Log.i(TAG, "s is " + s);
            if (expression.contains(s))
            {

                exp_index = expression.indexOf(s);
                if ((exp_index == 0) && (expression.substring(0,1).equals("-") || expression.substring(0,1).equals("+")))
                    break;
                if (exp_index < expression.length())
                    next_index = exp_index + 1;
                if(exp_index != 0)
                    prev_index = exp_index - 1;
                Log.i(TAG, " 1: " + expression.substring(exp_index, next_index).equals("-")
                        + " 2: " + (exp_index != 0)
                        + " 3: " + (exp_index != (expression.length()- 1))
                        + " 4: " +  ((next_index + 1) <= (expression.length() - 1))
                        + " expression.substring(exp_index, next_index) is " + expression.substring(exp_index, next_index));
                if(expression.substring(exp_index, next_index).equals("-") &&
                                        (exp_index != 0) &&
                                        ((exp_index != (expression.length()- 1)) &&
                                        ((next_index + 1) <= (expression.length() - 1)))) {

                    Log.i(TAG, "next string: " + expression.substring(next_index, next_index + 1)
                    + " next next string: " + expression.substring(next_index + 1, next_index + 2));
                    if (expression.substring(next_index, next_index + 1).equals("-") &&
                            expression.substring(next_index + 1, next_index + 2).equals("-")) {
                        expression = expression.substring(0, exp_index) + "/" + expression.substring(exp_index + 3);
                        Log.i(TAG, "/  detected! New exp: " + expression);
                    }
                }

            }

        }
        return expression;
    }
    private ArrayList<Rect> filterRectangles(ArrayList<Rect> rects) {
        double sum = 0.0;
        double mean = 0.0;
        for (int i = 0; i < rects.size(); i++) {
            sum += rects.get(i).height;
        }

        mean = sum / rects.size();
        Log.d(TAG, "Mean height = " + mean);

        for (int i = 0; i < rects.size(); i++) {
            if (rects.get(i).height < (mean - 30.0)) {
                Log.i(TAG, "Removed " + i + " because of its height: " + rects.get(i).height + "mean is: " + mean);
                rects.remove(i);
            }
        }
        return rects;
    }
}

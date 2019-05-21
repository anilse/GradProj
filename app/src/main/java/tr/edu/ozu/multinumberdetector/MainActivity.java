package tr.edu.ozu.multinumberdetector;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.StrictMode;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import javax.script.ScriptEngineManager;
import javax.script.ScriptEngine;
import javax.script.ScriptException;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Graduation_Project";
    public static final String[] symbols = {"+", "-", "x", "/"};
    private ImageProcessor mImgProcessor = new ImageProcessor();
    private FileUtils fileUtils = new FileUtils();
    ArrayList<String> mPhotoNames = new ArrayList<String>(50);
    private final int ACTIVITY_START_CAMERA_APP = 0;
    int mPhotoNum = 0;
    private String mImageFileLocation;

    //View Variables
    ImageView mPhoto;
    Button mNextButton;
    TextView mResultText;
    TextView mProbText;
    String resultText;
    String probText;
    String expression;
    double exp_result;
    int exp_index = 0;
    int next_index = 0;
    int prev_index = 0;
    boolean shouldCalculate = false;
    StringBuilder tempStringBuilder;

    ScriptEngineManager mgr = new ScriptEngineManager();
    ScriptEngine engine = mgr.getEngineByName("rhino");

    private Classifier mClassifier;
    private Executor executor = Executors.newSingleThreadExecutor();

    //Initialize open cv
    static {
        if(OpenCVLoader.initDebug()) {
            Log.d(TAG,"OpenCV Successfully Loaded");
        }
        else {
            Log.d(TAG,"OpenCV Load Not Successfully");
        }
    }

    private void init() {
        try {
            mClassifier = new Classifier(this);
        } catch (IOException e) {
            Toast.makeText(this, "failed_to_create_classifier", Toast.LENGTH_LONG).show();
            Log.e(TAG, "init(): Failed to create Classifier", e);
        }
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(tr.edu.ozu.multinumberdetector.R.layout.activity_main);
        //Uncomment this section to locally test openCV / detector interface without calling camera
        /*
        Bitmap origImage = BitmapFactory.decodeResource(this.getResources(), R.drawable.photo2);
        Bitmap bitmap = BitmapFactory.decodeResource(this.getResources(), R.drawable.photo2);

        Mat imgToProcess = mImgProcessor.preProcessImage(bitmap);
        Bitmap.createScaledBitmap(bitmap,imgToProcess.width(),imgToProcess.height(),false);
        Bitmap.createScaledBitmap(origImage,imgToProcess.width(),imgToProcess.height(),false);
        savePhoto(bitmap,"photo.jpg");
        Utils.matToBitmap(imgToProcess,bitmap);
        savePhoto(bitmap,"photo_preprocess.jpg");
        Mat boundImage = mImgProcessor.segmentAndDetect(imgToProcess,origImage,mDetector);
        Utils.matToBitmap(boundImage,bitmap);
        savePhoto(bitmap,"photo_bound.jpg");
        */
		init();
        mResultText = (TextView) findViewById(tr.edu.ozu.multinumberdetector.R.id.text_result);
        mProbText = (TextView) findViewById(tr.edu.ozu.multinumberdetector.R.id.text_prob);
        mPhoto = (ImageView) findViewById(tr.edu.ozu.multinumberdetector.R.id.photo);
        mNextButton = (Button) findViewById(tr.edu.ozu.multinumberdetector.R.id.next_button);

        //Next button in the app to slide between original photo,pre-processed photo and
        //result photo with boxes around digits

        mNextButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                switch(mPhotoNum) {
                    case 0 : resultText = "Original Photo"; probText = "0";
                            break;
                    case 1 : resultText = "Processed Photo"; probText = "1";
                            break;
                    case 2 :
                        shouldCalculate = false;
                        probText = mImgProcessor.getProbText();
                        int next;

                        for (String s : symbols) {
                            expression = mImgProcessor.getResultText();
                            if (expression.contains(s))
                            {

                                exp_index = expression.indexOf(s);
                                if ((exp_index == 0) && (expression.substring(0,1).equals("-") || expression.substring(0,1).equals("+")))
                                    break;
                                if (exp_index < expression.length())
                                    next_index = exp_index + 1;
                                if(exp_index != 0)
                                    prev_index = exp_index - 1;
                                if(expression.substring(exp_index, next_index).equals("-") &&
                                        (exp_index != 0) &&
                                        ((exp_index != (expression.length()- 1)) &&
                                                ((next_index + 1) <= (expression.length() - 1)))){
                                    if(expression.substring(next_index, next_index + 1).equals("-") &&
                                            expression.substring(next_index +1, next_index + 2).equals("-")){
                                        expression = expression.substring(0, exp_index) + "/" + expression.substring(exp_index+3);
                                        Log.i(TAG, "/  detected! New exp: " + expression );

                                    }

                                }
                                try {
                                   next = Integer.parseInt(expression.substring(next_index,next_index+1));
                                } catch (NumberFormatException e) {
                                    shouldCalculate = false;
                                    break;
                                }
                                //Log.i(TAG, "contains the symbol:" + s + "with index" + exp_index + "next: " + mImgProcessor.getResultText().substring(next_index) + " prev:" + mImgProcessor.getResultText().substring(prev_index, exp_index) );
                                if((Integer.parseInt(expression.substring(next_index, next_index +1)) < 10) &&
                                        (Integer.parseInt(expression.substring(prev_index, exp_index)) < 10)) {
                                    shouldCalculate = true;
                                    break;
                                }
                            }
                            else {
                                resultText = "There are only digits. Detections are: " + expression;
                            }
                        }
                        Log.i(TAG, "shouldcalculate is: " + shouldCalculate);
                        if (shouldCalculate){
                            String calculate = expression;
                            if(resultText.contains("x")){
                                calculate = expression.substring(0,exp_index)+'*'+expression.substring(exp_index+1);
                            }
                            try {
                                exp_result = (double) engine.eval(calculate);
                                resultText = "Detected expression: " + expression + " equals:" + exp_result;
                            } catch (ScriptException e) {
                                e.printStackTrace();
                            }
                        }
                        else
                            resultText ="No math expression detected. Detections are: " + expression;
                        break;

                }
                mPhoto.setImageBitmap(fileUtils.getSavedImage(mPhotoNames.get(mPhotoNum)));

                mResultText.setText(resultText);
                mProbText.setText(probText);
                    mPhotoNum += 1;
                mPhotoNum = mPhotoNum % mPhotoNames.size();
            }

        });
    }
    public void savePhoto(Bitmap bm, String photoName) {
        mPhotoNames.add(photoName);
        fileUtils.saveImage(bm,photoName);
    }
    //Once the photo is clicked inside the Camera App it comes to this task
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        if (requestCode == ACTIVITY_START_CAMERA_APP && resultCode == RESULT_OK) {

            //Get the photo clicked by the camera - reduced size to ensure does not make
            //the app crash in low memory situation - was a BUG initially which was not caught
            // when app was run on high memory Nexus 6p phone (64 GB). When app was re-run on Samsung
            // S5 with 32 GB it crashed dues to insufficient memory.
            Bitmap bitmap = fileUtils.getCameraPhoto("photo.jpg");
            Bitmap origImage = fileUtils.getCameraPhoto("photo.jpg");

            //Preprocess the image to remove noise, amplify the region of interest with the number
            //by making it bright white and make the background completely black.
            Mat imgToProcess = mImgProcessor.preProcessImage(bitmap);

            //Scale down the bitmap based on the processing height and width (640 x 480)
            Bitmap.createScaledBitmap(bitmap,imgToProcess.width(),imgToProcess.height(),false);
            Bitmap.createScaledBitmap(origImage,imgToProcess.width(),imgToProcess.height(),false);

            //Convert to bitmap and save the photo to display in the app.
            Utils.matToBitmap(imgToProcess.clone(),bitmap);
            savePhoto(bitmap,"photo_preprocess.jpg");

            //Pass the preprocessed image to perform segmentation and recognition. This also
            //overlays rectangular boxes on the segmented digits.
            Mat boundImage = mImgProcessor.segmentAndRecognize(imgToProcess,origImage,mClassifier);
            Utils.matToBitmap(boundImage.clone(),bitmap);
            savePhoto(bitmap,"photo_bound.jpg");
            mResultText.setText(mImgProcessor.getResultText());
        }
    }
    //This is the task called when the take Photo button is pressed.
    //Transfers control to Camera Application of the phone
    public void takePhoto(View view) {
        StrictMode.VmPolicy.Builder builder = new StrictMode.VmPolicy.Builder();
        StrictMode.setVmPolicy(builder.build());
        Intent callCameraApplicationIntent = new Intent();
        callCameraApplicationIntent.setAction(MediaStore.ACTION_IMAGE_CAPTURE);
        File photoFile = null;
        photoFile = fileUtils.createImageFile("photo.jpg");
        mImageFileLocation = photoFile.getAbsolutePath();
        mPhotoNames.add(photoFile.getName());
        Log.d(TAG,"Image name : " + photoFile.getName());
        callCameraApplicationIntent.putExtra(MediaStore.EXTRA_OUTPUT, Uri.fromFile(photoFile));
        startActivityForResult(callCameraApplicationIntent,ACTIVITY_START_CAMERA_APP);
    }
}

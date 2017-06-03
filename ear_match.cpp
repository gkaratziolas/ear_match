#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#define CANNY_THRESH 60

using namespace std;
using namespace cv;


/** Global variables */
String right_ear_cascade_name = "haar_right_ear.xml";
CascadeClassifier right_ear_cascade;
string window_name = "Capture - Ear detection";
RNG rng(12345);
Mat earMat;
vector<Point> saved_ear;
vector<Point> ear_hull;


void detectEars( Mat frame );
void enlargeRect(Rect &a, Mat frame);
int largestRect(vector<Rect> R);
void createEarDescriptor();

/** @function main */
int main( int argc, const char** argv )
{
    CvCapture* capture;
    Mat frame;
  
    //-- 1. Load the cascades
    if( !right_ear_cascade.load( right_ear_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    //-- 2. Read the video stream
    capture = cvCaptureFromCAM( 0 );
    if( capture )
    {
        while( true )
        {
            frame = cvQueryFrame( capture );

            //-- 3. Apply the classifier to the frame
            if( !frame.empty() )
            {
                detectEars( frame );
            }
            else
            {
                printf(" --(!) No captured frame -- Break!"); break;
            }
            int c = waitKey(10);
            if( (char)c == 'c' ) { break; }
            if( (char)c == 'a' ) {
                saved_ear = ear_hull;
            }
            // toggle pause on space key
            if ( (char)c == ' ' ){
                c = waitKey(10);
                while ( (char)c != ' ' ){
                    c = waitKey(10);
                    if( (char)c == 'a' ) {
                        saved_ear = ear_hull;
                    }
                }
            }

        }
    }
    return 0;
}


// enlarge a rectangle by 30% in each direction
// useful here to ensure none of the ear is left out of the frame
void enlargeRect(Rect &a, Mat frame)
{
    int width_increase  = 0.3*a.width;
    int height_increase = 0.3*a.height;
    a += cv::Point(-0.5*width_increase,-0.5*height_increase);
    a += cv::Size(width_increase, height_increase);
  
    if (a.x < 0) a.x = 0;
    if (a.y < 0) a.y = 0;
    if (a.x+a.width  > frame.cols) a.width  = frame.cols-a.x;
    if (a.y+a.height > frame.rows) a.height = frame.rows-a.y;
}

// finds the largest rectangle in a vector of Rect
// returns the index of the largest
// useful here for selecting the largest ear in frame
int largestRect(vector<Rect> R){
    int largest = 0;
    for (size_t i = 0; i < R.size(); i++){
        if ( i != largest ) {
            if ( (R[i].width * R[i].height) > ( R[largest].width * R[largest].height ) ){
                largest = i;
            }
        }
    }
    return largest;
}


/** @function detectAndDisplay */
void detectEars( Mat frame )
{
    std::vector<Rect> ears;
    Rect ear;
    Mat frame_gray;
  
    namedWindow("Right Ear", CV_WINDOW_AUTOSIZE);
  
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
  
    right_ear_cascade.detectMultiScale( frame_gray, ears, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    if ( ears.size() > 0 ){
        // find largest ear in the frame
        ear = ears[largestRect(ears)];

        // draw an ellipse around it
        Point center( ear.x + ear.width*0.5, ear.y + ear.height*0.5 );
        ellipse( frame, center, Size( ear.width*0.5, ear.height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
    
        // enlarge the selection area 
        enlargeRect(ear, frame);

        // crop ear from image
        Mat right_ear_ROI = frame_gray( ear );
        Mat croppedEar;
        right_ear_ROI.copyTo(croppedEar);
        imshow( "Right Ear", croppedEar);

        // resize the image to 200 by 200 pixels and blur
        resize(croppedEar, earMat, Size(200,200));
        GaussianBlur( earMat, earMat, Size( 9, 9), 0, 0 );

        // process ear Mat
        createEarDescriptor();
    }
    //-- Show what you got
    imshow( window_name, frame );
}

// comparison function object
bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1), false) );
    double j = fabs( contourArea(cv::Mat(contour2), false) );
    return ( i > j );
}

void createEarDescriptor(){
    Mat can, _img;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    double otsu_thresh_val = cv::threshold(
    earMat, _img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU
    );

    double high_thresh_val  = otsu_thresh_val,
       lower_thresh_val = otsu_thresh_val * 0.5;
    cv::Canny( earMat, can, lower_thresh_val, high_thresh_val );
    
    // Run canny edge detection on the Ear & display
    //Canny(earMat, can, CANNY_THRESH, CANNY_THRESH*2);
    namedWindow("Canny", WINDOW_AUTOSIZE);
    imshow("Canny", can);

    // Run a morphological close on the image to bridge any gaps
    //int morph_size = 5;
    //Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_size, morph_size));
    //morphologyEx(can, can, MORPH_CLOSE, element, Point(-1, -1), 1);
    //namedWindow("morph", WINDOW_AUTOSIZE);
    //imshow("morph", can);

    /// Find contours and convx hulls
    findContours( can, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    // requires two largest contours
    if (contours.size() > 1)
    {
        std::sort(contours.begin(), contours.end(), compareContourAreas);
        vector<vector<Point> >hull( contours.size() );

        for( int i = 0; i < contours.size(); i++ ){ 
            convexHull( Mat(contours[i]), hull[i], true );
        }


        Mat drawing = Mat::zeros( can.size(), CV_8UC3 );
        // join largest hull with next largest
        for( int i = 0; i < hull[1].size(); i++ ){ 
            hull[0].push_back(hull[1][i]);
        }

        ear_hull = hull[0];
        drawContours( drawing, hull, 0, Scalar(0,255,0), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );

        if ( saved_ear.size() > 0 ){
            double match = matchShapes(ear_hull, saved_ear,1,0.0);
            std::cout << match << std::endl;
            //drawContours( drawing, hull, largest_hull, Scalar(255,0,0), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
            polylines(drawing, saved_ear, true, Scalar(0,0,255), 1, 8);

        }
        namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
        imshow( "Contours", drawing );
    }

    /*
    for( int i = 0; i< contours.size(); i++ )
    {
        if ( arcLength(contours[i], true) > 100 ){
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            //drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
            //drawContours( drawing, contours, i, color,CV_FILLED);
            //color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            //drawContours( drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        }
    }*/

    /// Show in a window

}
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;

int main()
{
    /* SIFT FEATURES

    cv::Mat input = cv::imread("IMAG3157.jpg", 0);
    std::vector<cv::KeyPoint> keypoints[1000];
    cv::Mat descriptor[1000];
    cv::SiftFeatureDetector detector;
    cv::SiftDescriptorExtractor extractor;
    float siftLabels[1000];
    int counter = 0;
    cv::Mat croppedImageOuter;
    
    for(int i = 0; i < 7; i++) {
        for(int j = 0; j < 12; j++) {            
            // Transform it into the C++ cv::Mat format
            cv::Mat image(input); 

            // Setup a rectangle to define your region of interest
            cv::Rect myROI(300*i, 300*j, 300, 300);

            // Crop the full image to that image contained by the rectangle myROI
            // Note that this doesn't copy the data
            cv::Mat croppedImage = image(myROI);
            detector.detect(croppedImage, keypoints[counter]);
            extractor.compute( croppedImage, keypoints[counter], descriptor[counter] );
            reduce(descriptor[counter],descriptor[counter], 0, CV_REDUCE_AVG);
            siftLabels[counter] = 7;

            counter++;
        }
    }

    input = cv::imread("IMAG3163.jpg", 0);

    for(int i = 0; i < 7; i++) {
        for(int j = 0; j < 12; j++) {
            // Transform it into the C++ cv::Mat format
            cv::Mat image(input); 

            // Setup a rectangle to define your region of interest
            cv::Rect myROI(300*i, 300*j, 300, 300);

            // Crop the full image to that image contained by the rectangle myROI
            // Note that this doesn't copy the data
            cv::Mat croppedImage = image(myROI);
            detector.detect(croppedImage, keypoints[counter]);
            extractor.compute( croppedImage, keypoints[counter], descriptor[counter] );
            reduce(descriptor[counter],descriptor[counter], 0, CV_REDUCE_AVG);
            siftLabels[counter] = 5;

            counter++;

            croppedImageOuter = croppedImage;
        }
    }

    input = cv::imread("IMAG3191.jpg", 0);

    for(int i = 0; i < 7; i++) {
        for(int j = 0; j < 12; j++) {
            // Transform it into the C++ cv::Mat format
            cv::Mat image(input); 

            // Setup a rectangle to define your region of interest
            cv::Rect myROI(300*i, 300*j, 300, 300);

            // Crop the full image to that image contained by the rectangle myROI
            // Note that this doesn't copy the data
            cv::Mat croppedImage = image(myROI);
            detector.detect(croppedImage, keypoints[counter]);
            extractor.compute( croppedImage, keypoints[counter], descriptor[counter] );
            reduce(descriptor[counter],descriptor[counter], 0, CV_REDUCE_AVG);
            siftLabels[counter] = 2;

            counter++;
        }
    }


    input = cv::imread("IMAG3172.jpg", 0);

    for(int i = 0; i < 7; i++) {
        for(int j = 1; j < 12; j++) {
            // Transform it into the C++ cv::Mat format
            cv::Mat image(input); 

            // Setup a rectangle to define your region of interest
            cv::Rect myROI(300*i, 300*j, 300, 300);

            // Crop the full image to that image contained by the rectangle myROI
            // Note that this doesn't copy the data
            cv::Mat croppedImage = image(myROI);
            detector.detect(croppedImage, keypoints[counter]);
            extractor.compute( croppedImage, keypoints[counter], descriptor[counter] );
            reduce(descriptor[counter],descriptor[counter], 0, CV_REDUCE_AVG);
            siftLabels[counter] = 4;

            counter++;
        }
    }*/

    Mat src, src_gray;
  Mat grad;
  char* window_name = "Sobel Demo - Simple Edge Detector";
  int scale = 1;
  int delta = 0;
  int ddepth = CV_64F;
  int counter = 0;
  cv::Mat input;
  cv::Mat croppedImageOuter;
    
  int c;

  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  cv::Mat angles;
  Mat trainingDataMat(252, 20, CV_32FC1, double(0));
  Mat labelsMat(252, 1, CV_32FC1, double(0));
  int q = 0;
  //Mat siftLabels(1, 168, CV_32FC1, double(0));

  std::cout << "Hello before HOG" << "\n";

  input = imread( "IMAG3157.jpg" );
  std::cout << "Hello before G" << "\n";
  GaussianBlur( input, input, Size(3,3), 0, 0, BORDER_DEFAULT );
  std::cout << "Hello before C" << "\n";
  cvtColor( input, input, CV_BGR2GRAY );
  std::cout << "Hello after C" << "\n";
    for(int i = 0; i < 7; i++) {
        for(int j = 0; j < 12; j++) {     
        std::cout << "Hello from inside" << "," << i << " : " << j << "\n";       
            // Transform it into the C++ cv::Mat format
            cv::Mat image(input); 

            // Setup a rectangle to define your region of interest
            cv::Rect myROI(300*i, 300*j, 300, 300);

            // Crop the full image to that image contained by the rectangle myROI
            // Note that this doesn't copy the data
            cv::Mat croppedImage = image(myROI);
            
            /// Gradient X
            //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
            Sobel( croppedImage, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_x, abs_grad_x );
            /// Gradient Y
            //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
            Sobel( croppedImage, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_y, abs_grad_y );
  
            /*cv::phase(grad_x, grad_y, angles, true);
            std::cout << "Hello after all prep" << "\n";
            std::cout << angles.at<float>(1, 5) << " <- angle value"<< "," << i << " : " << j  << "\n";*/

            
            
            for (int r = 0; r < 300; r++)
            {
                for (int k = 0; k < 300; k++)
                {
                    float valueX = abs_grad_x.at<float>(r,k);
                    float valueY = abs_grad_y.at<float>(r,k);
                    // Calculate the corresponding single direction, done by applying the arctangens function
                    float result = fastAtan2(valueX,valueY);
                    if(isnan(result) || isinf(result)) {
                      std::cout << "Cooly" << "\n";
                    }

                    if(result != result) {
                        std::cout << "COOL" << "\n";
                        continue;
                    }


                    /*std::cout << "Angles value: " << angles.at<float>(r, k) << " || " << r << " : " << k << " Cell: " << i <<" : "<< j<<"\n";
                    */
                    q = (int)result/18;
                    if(q == 20) {
                        std::cout << "Twenty" << "\n";
                        q = 19;
                    }
                    std::cout << counter << " " << q << "\n"; 
                    trainingDataMat.at<float>(counter, q)++;

                    
                }
            }

            labelsMat.at<float>(counter, 0) = 7;
            counter++;
        }
    }
    std::cout << "Hello after HOG1" << "\n";
    

    input = imread( "IMAG3163.jpg" );
    GaussianBlur( input, input, Size(3,3), 0, 0, BORDER_DEFAULT );
    cvtColor( input, input, CV_BGR2GRAY );

    for(int i = 0; i < 7; i++) {
        for(int j = 0; j < 12; j++) {     
        std::cout << "Hello from inside" << "," << i << " : " << j << "\n";       
            // Transform it into the C++ cv::Mat format
            cv::Mat image(input); 

            // Setup a rectangle to define your region of interest
            cv::Rect myROI(300*i, 300*j, 300, 300);

            // Crop the full image to that image contained by the rectangle myROI
            // Note that this doesn't copy the data
            cv::Mat croppedImage = image(myROI);
            
            /// Gradient X
            //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
            Sobel( croppedImage, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_x, abs_grad_x );
            /// Gradient Y
            //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
            Sobel( croppedImage, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_y, abs_grad_y );
  
            //cv::phase(grad_x, grad_y, angles, true);
            //std::cout << "Hello after all prep" << "\n";
            //std::cout << angles.at<float>(1, 5) << " <- angle value"<< "," << i << " : " << j  << "\n";

            
            
            for (int r = 0; r < 300; r++)
            {
                for (int k = 0; k < 300; k++)
                {
                    float valueX = abs_grad_x.at<float>(r,k);
                    float valueY = abs_grad_y.at<float>(r,k);
                    // Calculate the corresponding single direction, done by applying the arctangens function
                    float result = fastAtan2(valueX,valueY);
                    if(result != result) {
                        std::cout << "COOL" << "\n";
                        continue;
                    }
                    //std::cout << "Angles value: " << angles.at<float>(r, k) << " || " << r << " : " << k << " Cell: " << i <<" : "<< j<<"\n";
                    
                    q = (int)result/18;
                    if(q == 20) {
                        std::cout << "Twenty" << "\n";
                        q = 19;
                    }
                    std::cout << counter << " " << q << "\n"; 
                    trainingDataMat.at<float>(counter, q)++;
                }
            }

            labelsMat.at<float>(counter, 0) = 5;
            counter++;
        }
    }

    input = imread( "IMAG3172.jpg" );
    GaussianBlur( input, input, Size(3,3), 0, 0, BORDER_DEFAULT );
    cvtColor( input, input, CV_BGR2GRAY );

    for(int i = 0; i < 7; i++) {
        for(int j = 2; j < 12; j++) {     
        std::cout << "Hello from inside" << "," << i << " : " << j << "\n";       
            // Transform it into the C++ cv::Mat format
            cv::Mat image(input); 

            // Setup a rectangle to define your region of interest
            cv::Rect myROI(300*i, 300*j, 300, 300);

            // Crop the full image to that image contained by the rectangle myROI
            // Note that this doesn't copy the data
            cv::Mat croppedImage = image(myROI);
            
            /// Gradient X
            //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
            Sobel( croppedImage, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_x, abs_grad_x );
            /// Gradient Y
            //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
            Sobel( croppedImage, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_y, abs_grad_y );
  
            //cv::phase(grad_x, grad_y, angles, true);
            //std::cout << "Hello after all prep" << "\n";
            //std::cout << angles.at<float>(1, 5) << " <- angle value"<< "," << i << " : " << j  << "\n";

            
            
            for (int r = 0; r < 300; r++)
            {
                for (int k = 0; k < 300; k++)
                {
                    float valueX = abs_grad_x.at<float>(r,k);
                    float valueY = abs_grad_y.at<float>(r,k);
                    // Calculate the corresponding single direction, done by applying the arctangens function
                    float result = fastAtan2(valueX,valueY);
                    if(result != result) {
                        std::cout << "COOL" << "\n";
                        continue;
                    }
                    //std::cout << "Angles value: " << angles.at<float>(r, k) << " || " << r << " : " << k << " Cell: " << i <<" : "<< j<<"\n";
                    
                    q = (int)result/18;
                    if(q == 20) {
                        std::cout << "Twenty" << "\n";
                        q = 19;
                    }
                    std::cout << counter << " " << q << "\n"; 
                    trainingDataMat.at<float>(counter, q)++;
                }
            }

            labelsMat.at<float>(counter, 0) = 2;
            counter++;
        }
    }
    
/*
    input = imread( "IMAG3172.jpg" );
    GaussianBlur( input, input, Size(3,3), 0, 0, BORDER_DEFAULT );
  cvtColor( input, input, CV_BGR2GRAY );
    for(int i = 0; i < 7; i++) {
        for(int j = 0; j < 12; j++) {     
        std::cout << "Hello from inside" << "," << i << " : " << j << "\n";       
            // Transform it into the C++ cv::Mat format
            cv::Mat image(input); 

            // Setup a rectangle to define your region of interest
            cv::Rect myROI(300*i, 300*j, 300, 300);

            // Crop the full image to that image contained by the rectangle myROI
            // Note that this doesn't copy the data
            cv::Mat croppedImage = image(myROI);
            
            /// Gradient X
            //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
            Sobel( croppedImage, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_x, abs_grad_x );
            /// Gradient Y
            //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
            Sobel( croppedImage, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_y, abs_grad_y );
  
            cv::phase(grad_x, grad_y, angles, true);
            std::cout << "Hello after all prep" << "\n";
            std::cout << angles.at<float>(1, 5) << " <- angle value"<< "," << i << " : " << j  << "\n";

            
            
            for (int r = 0; r < 300; r++)
            {
                for (int k = 0; k < 300; k++)
                {
                    float valueX = abs_grad_x.at<float>(r,k);
                    float valueY = abs_grad_y.at<float>(r,k);
                    // Calculate the corresponding single direction, done by applying the arctangens function
                    float result = fastAtan2(valueX,valueY);
                    if(result != result) {
                        std::cout << "COOL" << "\n";
                        continue;
                    }
                    std::cout << "Angles value: " << angles.at<float>(r, k) << " || " << r << " : " << k << " Cell: " << i <<" : "<< j<<"\n";
                    
                    q = (int)result/18;
                    std::cout << counter << " " << q << "\n"; 
                    v[counter][q]++;
                }
            }

            siftLabels[counter] = 7;
            counter++;
        }
    }

*/



  

    /// Total Gradient (approximate)
    //addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    //imshow( window_name, grad );
    //waitKey(0);
    
    std::cout << counter << "\n";
    //std::cout << " Type: "<< descriptor[1].type() << "\n";

    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    // Set up training data
    /*float labels[counter] = {1.0, -1.0, -1.0, -1.0};*/
   // Mat labelsMat(counter, 1, CV_32FC1, siftLabels);

    /*float trainingData[counter][128] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };*/
    /*Mat trainingDataMat(counter, 128, CV_32FC1);
   // Mat trainingDataMat(counter, 128, CV_32FC1, descriptor);
    //descriptor.convertTo();
    for(int r = 0; r < counter; r++)
    {
        for(int c = 0; c < 20; c++)
        {
            trainingDataMat.at<float>(r,c) = v[r][c];
            //std::cout << trainingDataMat.at<float>(r,c) << "\n";
        }
    }     */   

    std::cout << trainingDataMat.rows << " : " << trainingDataMat.cols << "\n";

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::EPS_SVR;
    params.kernel_type = CvSVM::RBF;
    //params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 30, 1e-6);
    params.C = 1e+01; 
    params.p = 0.001;

    std::cout << "Hello after params" << "\n";
    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
    SVM.save("svmTrainedModel");

    std::cout << "Hello after saving the file" << "\n"; 
    std::cout << croppedImageOuter.type() << "\n";

    float result = SVM.predict(trainingDataMat.row(24));
    std::cout << "Prediction is: " << result << "\n";

    /*Vec3b green(0,255,0), blue (255,0,0);
    // Show the decision regions given by the SVM
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = SVM.predict(sampleMat);

            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                 image.at<Vec3b>(i,j)  = blue;
        }

    // Show the training data
    int thickness = -1;
    int lineType = 8;
    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

    // Show support vectors
    thickness = 2;
    lineType  = 8;
    int c     = SVM.get_support_vector_count();

    for (int i = 0; i < c; ++i)
    {
        const float* v = SVM.get_support_vector(i);
        circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }

    imwrite("result.png", image);        // save the image

    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);*/

}

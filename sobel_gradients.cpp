#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

/** @function main */
int main( int argc, char** argv )
{

  Mat src, src_gray;
  Mat grad;
  char* window_name = "Sobel Demo - Simple Edge Detector";
  int scale = 1;
  int delta = 0;
  int ddepth = CV_64F;

  int c;

  /// Load an image
  src = imread( "terrain.jpg" );

  if( !src.data )
  { return -1; }

  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// Convert it to gray
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Create window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );
  cout << abs_grad_x.rows << " : " << abs_grad_x.cols << "\n";

  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );
  cout << abs_grad_y.rows << " : " << abs_grad_y.cols << "\n";
  cout << abs_grad_x.type() << "\n";
  cout << abs_grad_y.type() << "\n";

  cv::Mat angles;
  cv::phase(grad_x, grad_y, angles, true);
  cout << angles.rows << " : " << angles.cols << "\n";

  int q = 0;
  int v[20] = {0};
  for (int i = 0; i < angles.rows; i++)
  {
    for (int j = 0; j < angles.cols; j++)
    {
      q = angles.at<float>(i, j)/9;
      v[q]++;
    }
  }


  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  imshow( window_name, grad );

  waitKey(0);

  return 0;
  }

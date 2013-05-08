#ifndef __IMAGEWARP_H_
#define __IMAGEWARP_H_
#include "opencv2\opencv.hpp"

#include <stdio.h>
#include "struct.h"
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

Mat warp(Mat base,Mat warped,Mat H);


#endif
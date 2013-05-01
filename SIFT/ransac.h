#ifndef __RANSAC_H_
#define __RANSAC_H_
#include "opencv2\opencv.hpp"

#include <stdio.h>

#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
using namespace std;
using namespace cv;

void ransac(vector<KeyPoint> keypoint1,vector<KeyPoint> keypoint2,vector<vector<DMatch>>& matches,int max_iter,Mat H,double threshold);


#endif
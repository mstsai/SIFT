#ifndef __KNN_H_
#define __KNN_H_
#include "opencv2\opencv.hpp"

#include <stdio.h>

#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
using namespace std;
using namespace cv;
struct node{
	int index;
	float distance;
};
void knn(Mat& TrainData,Mat& quertData,int k,vector<vector<DMatch>>& matches);
bool comparator(node p,node q);

#endif
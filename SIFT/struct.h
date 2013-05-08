#ifndef __STRUCT_H_
#define __STRUCT_H_

#include <stdio.h>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
using namespace cv;

struct knnNeighbor{
	knnNeighbor(){
		imgidx=0;
		k=0;
	}
	int imgidx;            //original image feature point index
	int* neighbors;		   //knn result feature point index
	int k;
};
struct consensus_set{
	int trainindex;
	KeyPoint keypointt;
	int queryindex;
	KeyPoint keypointq;
	float distance;
};
#endif
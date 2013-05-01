#ifndef __STRUCT_H_
#define __STRUCT_H_

#include <stdio.h>

struct knnNeighbor{
	knnNeighbor(){
		imgidx=0;
		k=0;
	}
	int imgidx;            //original image feature point index
	int* neighbors;		   //knn result feature point index
	int k;
};

#endif
#include "ransac.h"
#include "opencv2\opencv.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
using namespace std;
using namespace cv;
//input keypoint1,keypoint2
//purpose:finding a matrix makes the most points  translate keypoint1 to keypoint2 
void ransac(vector<KeyPoint> keypoint1,vector<KeyPoint> keypoint2,vector<vector<DMatch>>& matches,int max_iter,Mat H,double threshold)
{
	srand(time(NULL));
	int rand,i=0;
	while(i<max_iter){

		i++;
	}
}
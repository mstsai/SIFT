#include "opencv2\opencv.hpp"
#include "imagewarp.h"
#include <cstdio>
#include <cstdlib>

#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "ransac.h"
#include "struct.h"

using namespace std;
using namespace cv;
extern FILE* fp;
Mat warp(Mat base,Mat warped,Mat H)
{
	Mat result(base);
	Mat points(3,1,CV_32F),transformed;
	fprintf(fp,"width:%d height:%d\n",result.rows,result.cols);
	for(int i=0;i<result.rows;i++){
		for(int j=0;i<result.cols;j++){
			points.at<float>(0,0)=(float)i;
			points.at<float>(1,0)=(float)j;
			points.at<float>(2,0)=1.0;
			transformed=H*points;
			transformed=transformed*(1/transformed.at<float>(2,0));
			fprintf(fp,"x:%d y:%d u:%.3f v:%.3f ratio:%.3f\n",i,j,transformed.at<float>(0,0),transformed.at<float>(1,0),transformed.at<float>(2,0));
			if((int)transformed.at<float>(0,0)<result.rows && (int)transformed.at<float>(1,0)<result.cols)
				result.at<Vec3b>(i,j)=warped.at<Vec3b>((int)abs(transformed.at<float>(0,0)),(int)abs(transformed.at<float>(1,0)));
			//result.at<Vec3b>(i,j)=
		}
	}
	return result;
}
#include "opencv2\opencv.hpp"
#include "imagewarp.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "ransac.h"
#include "struct.h"

using namespace std;
using namespace cv;
extern FILE* fp;
#define maximum 1000000.0
#define minimum -1000000.0
Mat warp(Mat warped,Mat scene,Mat H)
{
	//Mat result(base);
	Mat result(warped.size(),warped.type());
	//Mat result;
	Mat points(3,1,CV_32F),transformed;
	vector<Point2f> allpoints;
	Point2f p;
	fprintf(fp,"height:%d width:%d\n",result.rows,result.cols);
	print(H,H.rows,H.cols);
	float xMax=minimum,xMin=maximum,yMax=minimum,yMin=maximum;
	vector<Point2f> borders;
	int newborderx=0,newbordery=0,minx=0,miny=0;
	estimateSize(warped,scene,H,newborderx,newbordery,minx,miny);
	printf("new width:%d new height:%d\n",newbordery,newborderx);
	fprintf(fp,"new width:%d new height:%d\n",newbordery,newborderx);
	result.create(newborderx,newbordery,CV_8UC3);
	for(int i=0;i<warped.rows;i++){
		for(int j=0;j<warped.cols;j++){
				result.at<Vec3b>(i-miny,j-minx)=scene.at<Vec3b>(i,j);	
		}
	}
	//because x,y border is reverse
	//for scene image,position need to sub miny,minx
	//for warped image,with after inverse mapping,position need to add minx,miny
	for(int i=0;i<newborderx;i++){
		for(int j=0;j<newbordery;j++){
				points.at<float>(0,0)=(float)i;
				points.at<float>(1,0)=(float)j;
				points.at<float>(2,0)=1.0;
				transformed=H.inv()*points;
				transformed=transformed*(1/transformed.at<float>(2,0));
				//printf("x:%d y:%d u:%.3f v:%.3f ratio:%.3f fixedx:%d fixedy:%d\n",i,j,transformed.at<float>(0,0),transformed.at<float>(1,0),transformed.at<float>(2,0),round(transformed.at<float>(0,0)+minx),round(transformed.at<float>(1,0))+miny);

				//fprintf(fp,"x:%d y:%d u:%.3f v:%.3f ratio:%.3f fixedx:%d fixedy:%d\n",i,j,transformed.at<float>(0,0),transformed.at<float>(1,0),transformed.at<float>(2,0),round(transformed.at<float>(0,0)+minx),round(transformed.at<float>(1,0))+miny);
				//fflush(fp);
				if(round(transformed.at<float>(0,0)+minx)>0.0 && round(transformed.at<float>(0,0)+minx)<warped.rows && round(transformed.at<float>(0,0)+minx)<result.rows && round(transformed.at<float>(1,0)+miny)>0.0 && round(transformed.at<float>(1,0)+miny)<warped.cols && round(transformed.at<float>(1,0)+miny)<result.cols){
					//p.x=transformed.at<float>(0,0);
					//p.y=transformed.at<float>(1,0);
					//allpoints.push_back(p);
					if(result.at<Vec3b>(round(transformed.at<float>(0,0)+minx),round(transformed.at<float>(1,0)+miny))[0]==0 && result.at<Vec3b>(round(transformed.at<float>(0,0)+minx),round(transformed.at<float>(1,0)+miny))[1]==0 && result.at<Vec3b>(round(transformed.at<float>(0,0)+minx),round(transformed.at<float>(1,0)+miny))[2]==0)
					//if(result.at<Vec3b>(i,j)[0]==0 && result.at<Vec3b>(i,j)[1]==0 && result.at<Vec3b>(i,j)[2]==0)
						result.at<Vec3b>(i,j)=warped.at<Vec3b>(round(transformed.at<float>(0,0)+minx),round(transformed.at<float>(1,0)+miny));
					else
						result.at<Vec3b>(i,j)=warped.at<Vec3b>(round(transformed.at<float>(0,0)+minx),round(transformed.at<float>(1,0)+miny))/2+result.at<Vec3b>(i,j)/2;
							//result.at<Vec3b>(i,j)=base.at<Vec3b>(round(transformed.at<float>(0,0)),round(transformed.at<float>(1,0)))/2+result.at<Vec3b>(i,j)/2;
				}
				else{
						//result.at<Vec3b>(i,j)=0;
						//result.at<Vec3b>(round(transformed.at<float>(0,0)-minx),round(transformed.at<float>(1,0))-miny)=0;
					//result.at<Vec3b>(i,j)=0;
				}
			}
		}
		
		
	
	/*
	for(int i=0;i<result.rows;i++){
		for(int j=0;j<result.cols;j++){
			points.at<float>(0,0)=(float)i;
			points.at<float>(1,0)=(float)j;
			points.at<float>(2,0)=1.0;
			transformed=H.inv()*points;
			transformed=transformed*(1/transformed.at<float>(2,0));
			if(transformed.at<float>(0,0) < xMin)
				xMin=transformed.at<float>(0,0);
			if(transformed.at<float>(0,0) > xMax)
				xMax=transformed.at<float>(0,0);
			if(transformed.at<float>(1,0) < yMin)
				yMin=transformed.at<float>(1,0);
			if(transformed.at<float>(1,0) > yMax)
				yMax=transformed.at<float>(1,0);
			fprintf(fp,"x:%d y:%d u:%.3f v:%.3f ratio:%.3f\n",i,j,transformed.at<float>(0,0),transformed.at<float>(1,0),transformed.at<float>(2,0));
			if(((int)transformed.at<float>(0,0)<result.rows && transformed.at<float>(0,0)>0.0) && ( (int)transformed.at<float>(1,0)<result.cols && transformed.at<float>(1,0)>0.0)){
				//p.x=transformed.at<float>(0,0);
				//p.y=transformed.at<float>(1,0);
				//allpoints.push_back(p);
				
				result.at<Vec3b>(i,j)=base.at<Vec3b>((int)(transformed.at<float>(0,0)),(int)transformed.at<float>(1,0));
			}
			else{
				result.at<Vec3b>(i,j)=0;
			}
				
		}
	}
	fprintf(fp,"minx:%.3f miny:%.3f maxx:%.3f maxy:%.3f \n",xMin,yMin,xMax,yMax);
	*/
	//result.
	return result;
}
void estimateSize(Mat warped,Mat scene,Mat H,int& newborderx,int& newbordery,int& minX,int& minY)
{
	float minx=maximum,miny=maximum,maxx=minimum,maxy=minimum;
	Mat points(3,1,CV_32F),transformed;
	vector<Point2f> transformedPoint;
	Point2f p;
	
	for(int i=0;i<warped.rows;i++){
		for(int j=0;j<warped.cols;j++){
			points.at<float>(0,0)=(float)i;
			points.at<float>(1,0)=(float)j;
			points.at<float>(2,0)=1.0;
			transformed=H*points;
			transformed=transformed*(1/transformed.at<float>(2,0));
			if(transformed.at<float>(0,0) < minx)
				minx=transformed.at<float>(0,0);
			if(transformed.at<float>(0,0) > maxx)
				maxx=transformed.at<float>(0,0);
			if(transformed.at<float>(1,0) < miny)
				miny=transformed.at<float>(1,0);
			if(transformed.at<float>(1,0) > maxy)
				maxy=transformed.at<float>(1,0);
			//fprintf(fp,"x:%d y:%d u:%.3f v:%.3f ratio:%.3f\n",i,j,transformed.at<float>(0,0),transformed.at<float>(1,0),transformed.at<float>(2,0));
			p.x=transformed.at<float>(0,0);
			p.y=transformed.at<float>(1,0);
			transformedPoint.push_back(p);
		}
	}
	//fprintf(fp,"minx:%.3f miny:%.3f maxx:%.3f maxy:%.3f \n",minx,miny,maxx,maxy);
	
	/*
	minx=maximum;miny=maximum;maxx=minimum;maxy=minimum;
	for(int i=0;i<4;i++){
		switch(i){
			case 0:
				points.at<float>(0,0)=(float)0.0;
				points.at<float>(1,0)=(float)0.0;
				points.at<float>(2,0)=1.0;
				break;
			case 1:
				points.at<float>(0,0)=(float)warped.rows-1;
				points.at<float>(1,0)=(float)0.0;
				points.at<float>(2,0)=1.0;
				break;
			case 2:
				points.at<float>(0,0)=(float)0.0;
				points.at<float>(1,0)=(float)warped.cols-1;
				points.at<float>(2,0)=1.0;
				break;
			case 3:
				points.at<float>(0,0)=(float)warped.rows-1;
				points.at<float>(1,0)=(float)warped.cols-1;
				points.at<float>(2,0)=1.0;
				break;
		}
		transformed=H*points;
		if(transformed.at<float>(0,0) < minx)
			minx=transformed.at<float>(0,0);
		if(transformed.at<float>(0,0) > maxx)
			maxx=transformed.at<float>(0,0);
		if(transformed.at<float>(1,0) < miny)
			miny=transformed.at<float>(1,0);
		if(transformed.at<float>(1,0) > maxy)
			maxy=transformed.at<float>(1,0);
	}
	
	*/
	vector<float> xs,ys;
	
	xs.push_back(minx);
	xs.push_back(maxx);
	xs.push_back((float)scene.rows);
	xs.push_back(0);
	sort(xs.begin(),xs.end());
	ys.push_back(miny);
	ys.push_back(maxy);
	ys.push_back((float)scene.cols);
	ys.push_back(0.0);
	sort(ys.begin(),ys.end());
	minX=(int)xs.at(0);
	minY=(int)ys.at(0);
	newborderx=(int)(xs.at(xs.size()-1)-ys.at(0));
	newbordery=(int)(ys.at(xs.size()-1)-xs.at(0));
	//x,y border is reverse
	cout<<"size(a->b): "<<newborderx<<" "<<newbordery<<endl;
	fprintf(fp,"x:%.3f %.3f %.3f %.3f\n",xs.at(0),xs.at(1),xs.at(2),xs.at(3));
	fprintf(fp,"y:%.3f %.3f %.3f %.3f\n",ys.at(0),ys.at(1),ys.at(2),ys.at(3));
	fprintf(fp,"minx:%.3f miny:%.3f maxx:%.3f maxy:%.3f maxxval:%.3f minxval:%.3f maxyval:%.3f minyval:%.3f\n",minx,miny,maxx,maxy,xs.at(xs.size()-1),xs.at(0),ys.at(xs.size()-1),ys.at(0));
}
int round(float n)
{
	if(((float)ceil((double)n)+(float)floor((double)n))/2>n){
		return (int)n+1;
	}
	return (int)n;
}
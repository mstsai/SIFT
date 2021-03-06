#include "opencv2\opencv.hpp"
 
#include <stdio.h>
#include "knn.h"
#include "ransac.h"
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <windows.h>
using namespace std;
using namespace cv;

 FILE* fp;
 FILE* fin;
//
 void readFile(FILE*,int,vector<vector<DMatch>>);
int main(  )
{
    //source image
    char* img1_file = "DSC_0363.jpg";
    char* img2_file = "DSC_0364.jpg";
 
    // image read
    Mat tmp = cv::imread( img1_file, 1 );
    Mat in  = cv::imread( img2_file, 1 );
	printf("Height=%d \nWidth=%d \n", tmp.size().height, tmp.size().width);
    /* threshold      = 0.04;
       edge_threshold = 10.0;
       magnification  = 3.0;    */
 
    // SIFT feature detector and feature extractor
    cv::SiftFeatureDetector detector( 0.05, 5.0 );
    cv::SiftDescriptorExtractor extractor( 3.0 );
 
    // Feature detection
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
    detector.detect( tmp, keypoints1 );
    detector.detect( in, keypoints2 );
 
    // Feature display
    Mat feat1,feat2;
    drawKeypoints(tmp,keypoints1,feat1,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(in,keypoints2,feat2,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite( "feat1.bmp", feat1 );
    imwrite( "feat2.bmp", feat2 );
    int key1 = keypoints1.size();
    int key2 = keypoints2.size();
    printf("Keypoint1=%d \nKeypoint2=%d \n", key1, key2);
 
    // Feature descriptor computation
    Mat descriptor1,descriptor2,mask;
    extractor.compute( tmp, keypoints1, descriptor1 );
    extractor.compute( in, keypoints2, descriptor2 );

    printf("Descriptor1=(%d,%d) \nDescriptor2=(%d,%d)\n", descriptor1.size().height,descriptor1.size().width, descriptor2.size().height,descriptor2.size().width);
	//fp=fopen("tmp.txt","w");
	/*
	
	float val=0.0;
	for(int i=0;i<descriptor1.size().width;i++){
		val+=(descriptor1.at<float>(0,i)-descriptor2.at<float>(2,i))*(descriptor1.at<float>(0,i)-descriptor2.at<float>(2,i));
		
	}
	printf("%f\n",sqrt(val));
	*/
	LARGE_INTEGER ticksPerSecond;
	LARGE_INTEGER start_tick;
	LARGE_INTEGER end_tick;
	double elapsed;
	QueryPerformanceFrequency(&ticksPerSecond);
	QueryPerformanceCounter(&start_tick);

	int k=4;
	double threshold=100.0;
	int max_iter=100000;
	vector<vector<DMatch>> knnNeighbors;
	//knn(descriptor1,descriptor2,k,knnNeighbors);
	
	readFile(fin,k,knnNeighbors);
	QueryPerformanceCounter(&end_tick);
	elapsed = ((double)(end_tick.QuadPart - start_tick.QuadPart) / ticksPerSecond.QuadPart);
	cout<<"time:"<<elapsed<<" secs"<<endl;

	//test time
	Mat H;
	ransac(keypoints1,keypoints2,knnNeighbors,max_iter,H,threshold);


	/*
	for(int i=0;i<descriptor1.size().height;i++){
		for(int j=0;j<descriptor1.size().width;j++){
			//if(descriptor1.at<float>(i,j)>=180.0)
				fprintf(fp,"%.1f ",descriptor1.at<float>(i,j));
			if(j%8==7)
				fprintf(fp,"\n");
		}
		fprintf(fp,"\n === ===\n");
	}*/
	
	
	
	/*
	for(int i=0;i<descriptor2.size().height;i++){
		for(int j=0;j<descriptor2.size().width;j++){
			//if(descriptor2.at<float>(i,j)>=180.0)
				fprintf(fp,"%.1f ",descriptor2.at<float>(i,j));
		}
		fprintf(fp,"\n === ===\n");
	}*/
	

	// matching descriptors
	/*
	BruteForceMatcher<L2<float> > matcher;
	vector<DMatch> matches;
	matcher.match(descriptor1, descriptor2, matches);
	for(int i=0;i<matches.size();i++){
			//if(descriptor2.at<float>(i,j)>=180.0)
		fprintf(fp,"%.3f %d %d\n",matches.at(i).distance,matches.at(i).queryIdx,matches.at(i).trainIdx);
		//if(i%8==7)
			//fprintf(fp,"\n === ===\n");
	}
	// drawing the results
	namedWindow("matches", 1);
	Mat img_matches;
	drawMatches(tmp, keypoints1, in, keypoints2, matches, img_matches);
	imshow("matches", img_matches);
	waitKey(0);
	*/
	//Adding your code here
	//Including:
	
	//KNN
	//knn(descriptor1,descriptor2,)
	//RANSAC
	//Image Warping
	//Bonus (Gain/Blending/...)

    system("pause");
 
    return 0;
}
 void readFile(FILE* fins,int k,vector<vector<DMatch>> matches)
 {
	 vector<DMatch> oneMatch;
	 int counter=0,n=0;
	 char c;
	 oneMatch.resize(k);
	 fin=fopen("data.txt","r");
	 if(fin==NULL){
		 printf("no Data\n");
		 return;
	 }
	 while(true){

		 if(k==4){
			 n=fscanf(fin,"%d %d %f %d %f %d %f %d %f",&oneMatch.at(0).trainIdx,
				 &oneMatch.at(0).queryIdx,&oneMatch.at(0).distance,
				 &oneMatch.at(1).queryIdx,&oneMatch.at(1).distance,
				 &oneMatch.at(2).queryIdx,&oneMatch.at(2).distance,
				 &oneMatch.at(3).queryIdx,&oneMatch.at(3).distance
				 );
			 if(n!=k*2+1)
				 break;
			 matches.push_back(oneMatch);
			 
			 
		 }
		 //printf("%d ",++counter);
	 }
	 /*
	 for(int i=0;i<matches.size();i++){
		 printf("%d %d %f %d %f %d %f %d %f\n",
			 matches.at(i).at(0).trainIdx,
			 matches.at(i).at(0).queryIdx,matches.at(i).at(0).distance,
			 matches.at(i).at(1).queryIdx,matches.at(i).at(1).distance,
			 matches.at(i).at(2).queryIdx,matches.at(i).at(2).distance,
			 matches.at(i).at(3).queryIdx,matches.at(i).at(3).distance
			 );
	 }*///print info

 }
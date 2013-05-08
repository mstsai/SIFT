#ifndef __RANSAC_H_
#define __RANSAC_H_
#include "opencv2\opencv.hpp"

#include <stdio.h>
#include "struct.h"
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
using namespace std;
using namespace cv;
/*
vector<KeyPoint> keypoint1 first image's keypoint
vector<KeyPoint> keypoint2 second image's keypoint
int k  neighbor num
matches knn's result
int numofestimate random selected data number
int max_iter iterative times*/
void ransac(vector<KeyPoint> keypoint1,vector<KeyPoint> keypoint2,vector<vector<DMatch>>& matches,int k,int numofestimate,int max_iter,Mat& H,double percent,double threshold);
double compute(vector<KeyPoint> keypoint1,int* index,vector<KeyPoint> keypoint2,vector<int*> candidates,vector<vector<DMatch>>& matches,int n,int estimate,Mat& H,double percentage,double threshold,vector<consensus_set>& bestconsensus);
double leastSquare(vector<consensus_set> consensus,Mat H);
bool findinlier(vector<KeyPoint> keypoint1,vector<KeyPoint> keypoint2,vector<vector<DMatch>>& matches,Mat H,double percentage,double threshold,vector<consensus_set>& consensus);
Mat constructHomography(vector<KeyPoint> point1,vector<KeyPoint> point2);
Mat reestimate(vector<consensus_set>& consensus,int sampleNum);
void genAllCombination(int k,int estimate,vector<int*>& candidates);
void getrandNum(int* index,int size,int k);
void print(int* index,int k);
void print(Mat H,int rows,int cols);
void print(vector<consensus_set> consensus);
void print(vector<KeyPoint> keypoint1,int* index,int k);
void print(vector<KeyPoint> keyPoint);
#endif
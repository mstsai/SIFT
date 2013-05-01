#include "knn.h"
#include "opencv2\opencv.hpp"

#include <stdio.h>
#include <algorithm>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
using namespace std;
using namespace cv;

extern FILE* fp;
void knn(Mat& TrainData,Mat& queryData,int k,vector<vector<DMatch>>& matches)
{
	vector<DMatch> oneMatches;
	DMatch oneMatch;
	node* nodes=new node[queryData.size().height];

	matches.resize(TrainData.size().height);
	oneMatches.resize(k);

	float distance=0.0;
	Mat tmp;
	printf("train size:%d h:%d w:%d\n",TrainData.size(),TrainData.size().height,TrainData.size().width);
	for(int i=0;i<TrainData.size().height;i++){
		for(int j=0;j<queryData.size().height;j++){
			tmp=((TrainData.row(i)-queryData.row(j))*((TrainData.row(i)-queryData.row(j)).t()));
			nodes[j].distance=sqrt(tmp.at<float>(0,0));
			nodes[j].index=j;
			/*if( j==2){
				printf("distance:%f\n",nodes[2].distance);
				//printf("size:%d\n",sizeof(node)*queryData.size().height);
			}*/
			//printf("%d %d %d\n",tmp.size(),tmp.size().height,tmp.size().width);
		}
		sort(nodes,nodes+queryData.size().height,comparator);
		//fprintf(fp,"%d ",i);
		for(int m=0;m<k;m++){
			//fprintf(fp,"<%d %.3f>",nodes[m].index,nodes[m].distance);
			oneMatch.distance=nodes[m].distance;
			oneMatch.trainIdx=i;
			oneMatch.queryIdx=nodes[m].index;
			oneMatches.at(m)=oneMatch;
		}
		//fprintf(fp,"\n");
		
		matches.at(i)=oneMatches;
	
		
		
	}
	/*
	for(int i=0;i<TrainData.size().height;i++){
		fprintf(fp,"%d %d %.3f %d %.3f %d %.3f %d %.3f\n",
			matches.at(i).at(0).trainIdx,
			matches.at(i).at(0).queryIdx,matches.at(i).at(0).distance,
			matches.at(i).at(1).queryIdx,matches.at(i).at(1).distance,
			matches.at(i).at(2).queryIdx,matches.at(i).at(2).distance,
			matches.at(i).at(3).queryIdx,matches.at(i).at(3).distance
		);
	}*/
	delete nodes;
}
bool comparator(node p,node q){
	return (p.distance < q.distance);
}
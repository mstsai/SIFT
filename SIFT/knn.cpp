#include "knn.h"
#include "struct.h"
#include "opencv2\opencv.hpp"

#include <stdio.h>
#include <algorithm>
#include <Windows.h>
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
	//node tmpnode;
	printf("train size:%d h:%d w:%d\n",TrainData.size(),TrainData.size().height,TrainData.size().width);
	LARGE_INTEGER ticksPerSecond;
	LARGE_INTEGER start_tick;
	LARGE_INTEGER end_tick;
	QueryPerformanceFrequency(&ticksPerSecond);
	
	double elapsed=0.0,all_elapsed=0.0;
	for(int i=0;i<TrainData.size().height;i++){
		//QueryPerformanceCounter(&start_tick);
		
		for(int j=0;j<queryData.size().height;j++){

			tmp=((TrainData.row(i)-queryData.row(j))*((TrainData.row(i)-queryData.row(j)).t()));
			
			nodes[j].distance=tmp.at<float>(0,0);
			nodes[j].index=j;
			if(j>=k){
				int pos=k;
				for(int l=3;l>=0;l--){
					if(nodes[j].distance<nodes[l].distance)
						pos=l;
				}
				if(pos!=4)
					swap(nodes[pos],nodes[j]);
			}
			/*if( j==2){
				printf("distance:%f\n",nodes[2].distance);
				//printf("size:%d\n",sizeof(node)*queryData.size().height);
			}*/
			//printf("%d %d %d\n",tmp.size(),tmp.size().height,tmp.size().width);
		}
		
		
		//sort(nodes,nodes+queryData.size().height,comparator);
		//QueryPerformanceCounter(&end_tick);
		//elapsed = ((double)(end_tick.QuadPart - start_tick.QuadPart) / ticksPerSecond.QuadPart);
		//all_elapsed+=elapsed;
		//cout<<"time:"<<elapsed<<" secs"<<endl;
		//cout<<"all time:"<<all_elapsed<<" secs"<<endl;
		//fprintf(fp,"%d ",i);
		for(int m=0;m<k;m++){
			//fprintf(fp,"<%d %.3f>",nodes[m].index,nodes[m].distance);

			oneMatch.distance=sqrt(nodes[m].distance);
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
void swap(node& p,node& q)
{
	node tmp;
	tmp=p;p=q;q=tmp;
}
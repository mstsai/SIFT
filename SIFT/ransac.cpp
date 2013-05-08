
#include "opencv2\opencv.hpp"
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "ransac.h"
#include "struct.h"

using namespace std;
using namespace cv;
//input keypoint1,keypoint2
//purpose:finding a matrix makes the most points  translate keypoint1 to keypoint2 
#define MAXLENGTH 10000000
extern FILE* fp;
float nums[]={3.0,2.0,2.0,2.0,3.0,-2.0};
void ransac(vector<KeyPoint> keypoint1,vector<KeyPoint> keypoint2,vector<vector<DMatch>>& matches,int k,int numofestimate,int max_iter,Mat& H,double percent,double threshold)
{
	srand(time(NULL));
	//srand(1);
	int i=0,j=0;
	int *indexs=new int[k];
	double result=0.0;
	vector<int*> candidates;
	vector<consensus_set> consensus;
	/*
	Mat U(2,3,CV_32F),w,u,vt;
	
	U.at<float>(0,0)=3.0;
	U.at<float>(0,1)=2.0;
	U.at<float>(0,2)=2.0;
	U.at<float>(1,0)=2.0;
	U.at<float>(1,1)=3.0;
	U.at<float>(1,2)=-2.0;
	*/
	while(i<50){
		getrandNum(indexs,keypoint1.size(),k);
		genAllCombination(k,numofestimate,candidates);
		printf("size:%d %d i:%d\n",keypoint2.size(),matches.size(),i);
		/*
		SVD svd;
		svd.compute(U,w,u,vt,SVD::FULL_UV);
		print(w,w.rows,w.cols);
		print(u,u.rows,u.cols);
		print(vt,vt.rows,vt.cols);
		*/
		result=compute(keypoint1,indexs,keypoint2,candidates,matches,k,numofestimate,H,percent,threshold,consensus);
		//print(keypoint1,indexs,k);
		i++;
		if(result!=0.0){
			/*
			ransacmatches.resize(consensus.size());
			for(int i=0;i<consensus.size();i++){
				ransacmatches.at(i).resize(1);
				ransacmatches.at(i).at(0).distance=consensus.at(i).distance;
				ransacmatches.at(i).at(0).trainIdx=consensus.at(i).trainindex;
				ransacmatches.at(i).at(0).queryIdx=consensus.at(i).queryindex;
			}*/
			printf("end\n");
			break;
		}
			
	}
	delete indexs;
}

double compute(vector<KeyPoint> keypoint1,int* index,vector<KeyPoint> keypoint2,vector<int*> candidates,vector<vector<DMatch>>& matches,int n,int estimate,Mat& H,double percentage,double threshold,vector<consensus_set>& bestconsensus)
{
	int times=(int)pow((float)n,estimate);
	vector<KeyPoint> point1,point2;
	bool hasinlier=false;
	double result;
	vector<consensus_set> consensus;
	for(int i=0;i<times;i++){
		point1.clear();
		point2.clear();
		consensus.clear();
		for(int j=0;j<n;j++){
			//fprintf(fp,"index:%d point<x:%.3f y:%.3f>\n",index[j],keypoint1.at(index[j]).pt.x,keypoint1.at(index[j]).pt.y);
			KeyPoint tmp1=keypoint1.at(index[j]);
			//index[i] random selected point in KeyPoint1
			point1.push_back(tmp1);
			//fprintf(fp,"candidate: %d index2:%d point2<x:%.3f y:%.3f>\n",candidates.at(i)[j],matches.at(index[j]).at(candidates.at(i)[j]).queryIdx,
				//keypoint2.at(matches.at(index[j]).at(candidates.at(i)[j]).queryIdx).pt.x,keypoint2.at(matches.at(index[j]).at(candidates.at(i)[j]).queryIdx).pt.y);
			KeyPoint tmp2=keypoint2.at(matches.at(index[j]).at(candidates.at(i)[j]).queryIdx);
			//tmp2 = one of k points match KeyPoint1 in knn result
			point2.push_back(tmp2);
		}
		H=constructHomography(point1,point2);
		//fprintf(fp,"svd\n");
		//print(H,H.rows,H.cols);
		/*
		for(int j=0;j<n;j++){
			fprintf(fp,"index1:%d point<x:%.3f y:%.3f>\n",index[j],point1.at(j).pt.x,point1.at(j).pt.y);
			fprintf(fp,"index2:%d point<x:%.3f y:%.3f>\n",index[j],point2.at(j).pt.x,point2.at(j).pt.y);
		}*/
		hasinlier=findinlier(keypoint1,keypoint2,matches,H,percentage,threshold,consensus);
		print(consensus);
		if(hasinlier){
			/*H=reestimate(consensus,10*estimate);
			print(H,H.rows,H.cols);
			print(keypoint1);
			leastSquare(consensus,H);
			*/
			bestconsensus=consensus;
			return 1.0;
		}
	}

	
	return 0.0;
}
bool findinlier(vector<KeyPoint> keypoint1,vector<KeyPoint> keypoint2,vector<vector<DMatch>>& matches,Mat H,double percentage,double threshold,vector<consensus_set>& consensus)
{
	int neighborsNum=matches.at(0).size();
	double ratio=0.0;
	//fprintf(fp,"least square\n");
	//printf("%d %d\n",keypoint1.size(),neighborsNum);
	float u=0.0,v=0.0,x=0.0,y=0.0,bestu=0.0,bestv=0.0,mindistance=MAXLENGTH;
	int leastinlierNum=keypoint1.size()*percentage;
	int inlierNum=0;
	int queryindex=0;
	//u,v image 2's point corresponding to x,y
	//x,y image 1's point
	Mat point1(3,1,CV_32F);
	//matrix of [x,y,1]'
	Mat HmulPoint(3,1,CV_32F);
	//result of H*[x,y,1]'
	
	vector<KeyPoint> inlierp1;
	vector<KeyPoint> inlierp2;
	KeyPoint bestpoint;
	consensus_set matchpoint;
	//print(H,H.rows,H.cols);
	for(int i=0;i<keypoint1.size();i++){
		mindistance=MAXLENGTH;
		x=keypoint1.at(i).pt.x;
		y=keypoint1.at(i).pt.y;
		point1.at<float>(0,0)=x;
		point1.at<float>(1,0)=y;
		point1.at<float>(2,0)=1;
		HmulPoint=H*point1;
		ratio=HmulPoint.at<float>(2,0);
		/*
		if(i==0 || i==1){
			fprintf(fp,"x:%.3f y:%.3f ratio:%.3f\n",x,y,ratio);
			print(HmulPoint,HmulPoint.rows,HmulPoint.cols);
		}*/
		for(int j=0;j<neighborsNum;j++){
			u=keypoint2.at(matches.at(i).at(j).queryIdx).pt.x;
			v=keypoint2.at(matches.at(i).at(j).queryIdx).pt.y;
			float distance=sqrt((HmulPoint.at<float>(0,0)/ratio-u)*(HmulPoint.at<float>(0,0)/ratio-u)+(HmulPoint.at<float>(1,0)/ratio-v)*(HmulPoint.at<float>(1,0)/ratio-v));
			//float distance=sqrt((HmulPoint.at<float>(0,0)-u)*(HmulPoint.at<float>(0,0)-u)+(HmulPoint.at<float>(1,0)-v)*(HmulPoint.at<float>(1,0)-v));
			/*if(i==0 || i==1){
				fprintf(fp,"x:%.3f y:%.3f h.x/ratio:%.3f h.y/ratio:%.3f distance:%.3f\n",u,v,HmulPoint.at<float>(0,0)/ratio,HmulPoint.at<float>(1,0)/ratio,distance);
			}*/
			if(distance<mindistance){
				bestu=u;
				bestv=v;
				bestpoint=keypoint2.at(matches.at(i).at(j).queryIdx);
				queryindex=matches.at(i).at(j).queryIdx;
				mindistance=distance;		
				//fprintf(fp,"u:%.3f v:%.3f dis:%3f\n",u,v,distance);
			}
			
		}
		if(mindistance<(float)threshold){
			matchpoint.keypointt=keypoint1.at(i);
			matchpoint.trainindex=i;
			matchpoint.keypointq=bestpoint;
			matchpoint.queryindex=queryindex;
			matchpoint.distance=mindistance;
			consensus.push_back(matchpoint);
			inlierNum++;
		}
		if(keypoint1.size()-i+inlierNum<leastinlierNum){
			consensus.clear();
			return false;
		}//impossible to match
		//if(inlierNum>leastinlierNum)
			//fprintf(fp,"data%d:end inlierNum:%d\n",i,inlierNum);
	}
	//if(inlierNum>leastinlierNum/2)
		//fprintf(fp,"inlierNum:%d\n",inlierNum);
	if(inlierNum>leastinlierNum){
		printf("inlier>threshold\n");
		fprintf(fp,"inlierNum:%d\n",inlierNum);
		print(H,H.rows,H.cols);
		print(keypoint1);
		print(keypoint2);
		print(consensus);
		//reestimate
		return true;
	}
	else{
		consensus.clear();
		return false;
	}
}
double leastSquare(vector<consensus_set> consensus,Mat H)
{
	int sizes=consensus.size();
	float u=0.0,v=0.0,x=0.0,y=0.0,ratio=0.0;
	Mat point1(3,1,CV_32F);
	Mat HmulPoint(3,1,CV_32F);
	float meandistance=0.0;
	float std_dev=0.0,squares=0.0;
	for(int i=0;i<sizes;i++){
		x=consensus.at(i).keypointt.pt.x;
		y=consensus.at(i).keypointt.pt.y;
		u=consensus.at(i).keypointq.pt.x;
		v=consensus.at(i).keypointq.pt.y;
		point1.at<float>(0,0)=x;
		point1.at<float>(1,0)=y;
		point1.at<float>(2,0)=1;
		HmulPoint=H*point1;
		ratio=HmulPoint.at<float>(2,0);
		float distance=sqrt((HmulPoint.at<float>(0,0)/ratio-u)*(HmulPoint.at<float>(0,0)/ratio-u)+(HmulPoint.at<float>(1,0)/ratio-v)*(HmulPoint.at<float>(1,0)/ratio-v));
		meandistance+=distance;
		squares=squares+distance*distance;
		fprintf(fp,"x:%.3f y:%.3f u:%.3f v:%.3f dis:%.3f dis2:%.3f\n",x,y,u,v,distance,squares);
		
	}
	meandistance/=consensus.size();
	std_dev=sqrt(squares-(float)consensus.size()*meandistance*meandistance);
	fprintf(fp,"mean:%.3f std:%.3f all:%.3f\n",meandistance,std_dev,squares);
	return 0.0;
}
Mat reestimate(vector<consensus_set>& consensus,int sampleNum)
{
	Mat H;
	printf("sampleNum:%d\n",sampleNum);
	int* index=new int[sampleNum];
	int size=consensus.size();
	getrandNum(index,size,sampleNum);
	vector<KeyPoint> point1,point2;
	point1.resize(sampleNum);
	point2.resize(sampleNum);
	float meandistance=0.0;
	float std_dev=0.0,squares=0.0;
	for(int k=0;k<consensus.size();k++){
		meandistance+=consensus.at(k).distance;
		squares=squares+consensus.at(k).distance*consensus.at(k).distance;
		fprintf(fp,"dis:%.3f dis2:%.3f\n",meandistance,squares);
	}
	meandistance/=consensus.size();
	std_dev=sqrt(squares-(float)consensus.size()*meandistance*meandistance);
	fprintf(fp,"mean:%.3f std:%.3f all:%.3f\n",meandistance,std_dev,squares);
	for(int i=0;i<sampleNum;i++){
		point1.at(i)=consensus.at(index[i]).keypointt;
		point2.at(i)=consensus.at(index[i]).keypointq;
	}
	printf("size:%d\n",point1.size());
	H=constructHomography(point1,point2);
	delete index;
	return H;
}
void genAllCombination(int k,int estimate,vector<int*>& candidates)
{
	
	candidates.resize((int)pow((float)k,estimate));
	for(int i=0;i<candidates.size();i++){
		candidates.at(i)=new int[k];
		for(int j=0;j<k;j++)
			candidates.at(i)[j]=0;
	}
	
	//fprintf(fp,"%d %d\n",k,estimate);
	for(int i=0;i<candidates.size();i++){
		for(int j=0;j<k;j++){
			if(i>(int)pow((float)k,j)){
				if(j==0)
					candidates.at(i)[j]=i%k;
				else
					candidates.at(i)[j]=(i/(int)pow((float)k,j))%k;
			}
			else if(i==(int)pow((float)k,j)){
				candidates.at(i)[j]=1;
			}
			//fprintf(fp,"%d",candidates.at(i)[j]);
		}
		//fprintf(fp,"\n");
	}
}
Mat constructHomography(vector<KeyPoint> point1,vector<KeyPoint> point2)
{
	
	int rows=point1.size();
	//Mat U(2*rows,9,CV_32F);
	Mat U(2*rows,8,CV_32F);
	Mat H(3,3,CV_32F),w,u,vt,v,eigenval,eigenvec,b(2*rows,1,CV_32F),sol;
	float val=0.0;
	
	
	for(int i=0;i<rows;i++){
		U.at<float>(2*i,0)=point1.at(i).pt.x;
		U.at<float>(2*i,1)=point1.at(i).pt.y;
		U.at<float>(2*i,2)=1.0;
		U.at<float>(2*i,3)=U.at<float>(2*i,4)=U.at<float>(2*i,5)=0.0;
		U.at<float>(2*i,6)=-point1.at(i).pt.x*point2.at(i).pt.x;
		U.at<float>(2*i,7)=-point1.at(i).pt.y*point2.at(i).pt.x;
		//U.at<float>(2*i,8)=-point2.at(i).pt.x;
		U.at<float>(2*i+1,0)=U.at<float>(2*i+1,1)=U.at<float>(2*i+1,2)=0.0;
		U.at<float>(2*i+1,3)=point1.at(i).pt.x;
		U.at<float>(2*i+1,4)=point1.at(i).pt.y;
		U.at<float>(2*i+1,5)=1.0;
		U.at<float>(2*i+1,6)=-point1.at(i).pt.x*point2.at(i).pt.y;
		U.at<float>(2*i+1,7)=-point1.at(i).pt.y*point2.at(i).pt.y;
		//U.at<float>(2*i+1,8)=-point2.at(i).pt.y;
		b.at<float>(2*i,0)=point2.at(i).pt.x;
		b.at<float>(2*i+1,0)=point2.at(i).pt.y;
	}
	SVD svd;
	svd.compute(U,w,u,vt,SVD::FULL_UV);
	Mat diagonal=Mat::zeros(u.rows,vt.rows,CV_32F);
	for(int i=0;i<vt.rows;i++){
		diagonal.at<float>(i,i)=w.at<float>(i,0);
	}
	
	sol=((u*(diagonal)*vt).inv(DECOMP_SVD))*b;
	//U*x=b ,Ut*U*x=Ut*b,x=(Ut*U)-1*Ut*b=(v*(w)-1*ut)*b,if U=u*w*vt by svd;
	//sol=((U.t()*U).inv())*U.t()*b;
	//least square estimation

	/*for(int i=0;i<leastSquaresol.rows;i++){
		val+=leastSquaresol.at<float>(i,0)*leastSquaresol.at<float>(i,0);
	}
	leastSquaresol=leastSquaresol*(1/sqrt(val));*/
	/*
	fprintf(fp,"least square sol\n");
	print(leastSquaresol,leastSquaresol.rows,leastSquaresol.cols);
	fprintf(fp,"svd\n");
	print(vt,vt.rows,vt.cols);
	Mat UTU=U.t()*U;
	eigen(UTU,eigenval,eigenvec);
	fprintf(fp,"eigenval\n");
	print(eigenval,eigenval.rows,eigenval.cols);
	fprintf(fp,"eigenvec\n");
	print(eigenvec,eigenvec.rows,eigenvec.cols);
	*/
	//svd.compute(U,w,u,vt);
	//print(U,2*rows,9);
	//print(UTU,UTU.rows,UTU.cols);
	//print(u,u.rows,u.cols);
	//print(w,w.rows,w.cols);
	//fprintf(fp,"v");
	//print(vt,vt.rows,vt.cols);
	//printf("U:%d %d w:%d %d u:%d %d vt:%d %d\n",U.rows,U.cols,w.rows,w.cols,u.rows,u.cols,vt.rows,vt.cols);
	
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			if(i*3+j==8){
				H.at<float>(i,j)=1.0;
				//H.at<float>(i,j)=vt.at<float>(vt.rows-1,i*3+j);
				//H.at<float>(i,j)=vt.at<float>(i*3+j,vt.cols-1);
				//val+=H.at<float>(i,j)*H.at<float>(i,j);
			}
			else{
				//H.at<float>(i,j)=vt.at<float>(vt.rows-1,i*3+j);
				//H.at<float>(i,j)=vt.at<float>(i*3+j,vt.cols-1);
				//val+=H.at<float>(i,j)*H.at<float>(i,j);
				H.at<float>(i,j)=sol.at<float>(i*3+j,0);
			}
		}
	}//last column of vt is the solution of Ax=0
	//val=sqrt(val);
	//H=H*(1/val);
	//print(H,H.rows,H.cols);
	return H;
}
void getrandNum(int* index,int size,int k)
{
	int* unique=new int[size];
	for(int i=0;i<size;i++){
		unique[i]=0;
	}
	for(int i=0;i<k;i++){
		index[i]=rand()%size;
		while(unique[index[i]]==1){
			index[i]=rand()%size;
		}
		unique[index[i]]=1;
	}
	delete unique;
}
void print(Mat H,int rows,int cols)
{
	fprintf(fp,"Mat:\n");
	//printf("%.3f ",H.at<float>(i,j));
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			fprintf(fp,"%.3f ",H.at<float>(i,j));
			//printf("%.3f ",H.at<float>(i,j));
		}
		fprintf(fp,"\n");
		//printf("\n");
	}
}
void print(vector<consensus_set> consensus)
{
	for(int i=0;i<consensus.size();i++){
		fprintf(fp,"%.3f %.3f %.3f %.3f %.3f\n",consensus.at(i).keypointt.pt.x,consensus.at(i).keypointt.pt.y,
											consensus.at(i).keypointq.pt.x,consensus.at(i).keypointq.pt.y,
											consensus.at(i).distance);
		//fprintf(fp,"point:<x:%.3f y:%.3f> index:%d\n",keyPoint.at(i).pt.x,keyPoint.at(i).pt.y,i);
	}
}
void print(vector<KeyPoint> keyPoint,int* index,int k)
{
	for(int i=0;i<k;i++){
		printf("point:<x:%.3f y:%.3f> index:%d\n",keyPoint.at(index[i]).pt.x,keyPoint.at(index[i]).pt.y,index[i]);
	}
}
void print(vector<KeyPoint> keyPoint)
{
	for(int i=0;i<keyPoint.size();i++){
		fprintf(fp,"point:<x:%.3f y:%.3f> index:%d\n",keyPoint.at(i).pt.x,keyPoint.at(i).pt.y,i);
	}
}
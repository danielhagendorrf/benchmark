#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "C:/Users/Daniel Hagendorf/Documents/Visual Studio 2015/Projects/Project6/Project6/LBPRecognition.cpp"
#include "C:/Users/Daniel Hagendorf/Documents/Visual Studio 2015/Projects/Project6/Project6/LBPRecognition.h"
#include "copyFace.h"
#include "colorBalancing.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>

using namespace std;
using namespace cv;

//int eigen(Mat img, CascadeClassifier face_cascade, Ptr<BasicFaceRecognizer> model);
//Ptr<BasicFaceRecognizer> train(vector<Mat>& images, vector<int>& labels);
int LBP(Mat img, CascadeClassifier face_cascade, Ptr<FaceRecognizer> model);
Ptr<FaceRecognizer> trainLBP(vector<Mat>& images, vector<int>& labels);
void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator, CascadeClassifier face_cascade);
Mat copyFace(Mat img, int leftWidth, int bottomHeight, int rightWidth, int topHeight);
void colorBalancing(Mat& img, Mat& rImg, float percent);


String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

/** @function main */
int main(void)
{
	int tpr=0, fpr=0,tnr=0,fnr=0;

	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); return -1; };
	string csvTrain = string("c:/csv2.csv");
	string csvTest = string("c:/csv.csv");
	vector<Mat> imagesTrain;
	vector<int> labelsTrain;
	vector<Mat> imagesTest;
	vector<int> labelsTest;
	read_csv(csvTrain, imagesTrain, labelsTrain, ';', face_cascade);
	read_csv(csvTest, imagesTest, labelsTest, ';', face_cascade);
	Mat imgInstance;
	for (int i = 0; i < imagesTrain.size(); i++) {
		colorBalancing(imagesTrain[i], imagesTrain[i], 20.0f);
		std::vector<Rect> faces;
		//cvtColor(imagesTrain[i], imgInstance, COLOR_BGR2GRAY);
		equalizeHist(imgInstance, imgInstance);
		//-- Detect faces
		face_cascade.detectMultiScale(imgInstance, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		for (size_t j = 0; j < faces.size(); j++)
		{
			Point center(faces[j].x + faces[j].width / 2, faces[j].y + faces[j].height / 2);
			Mat faceROI = imagesTrain[i](faces[j]);
			imagesTrain[i] = copyFace(imagesTrain[i], faces[j].x, faces[j].y, faces[j].x + faces[j].width, faces[j].y + faces[j].height);
			//cvtColor(imagesTrain[i], imagesTrain[i], COLOR_BGR2GRAY);
		}
	}
	Ptr<FaceRecognizer> model=trainLBP(imagesTrain, labelsTrain);
	for (int i = 0; i < imagesTest.size(); i++) {
		colorBalancing(imagesTest[i], imagesTest[i], 20.0f);
		std::vector<Rect> faces;
		//cvtColor(imagesTest[i], imagesTest[i], COLOR_BGR2GRAY);
		equalizeHist(imagesTest[i], imagesTest[i]);
		face_cascade.detectMultiScale(imagesTest[i], faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		for (size_t j = 0; j < faces.size(); j++)
		{
			Point center(faces[j].x + faces[j].width / 2, faces[j].y + faces[j].height / 2);
			Mat faceROI = imagesTest[i](faces[j]);
			imagesTest[i] = copyFace(imagesTest[i], faces[j].x, faces[j].y, faces[j].x + faces[j].width, faces[j].y + faces[j].height);
			//cvtColor(imagesTest[i], imagesTest[i], COLOR_BGR2GRAY);
		}
		int prediction = LBP(imagesTest[i], face_cascade,model);
		if (labelsTest[i] == prediction && prediction == 0)
			tpr++;
		else
			if (labelsTest[i] != prediction && labelsTest[i] == 0)
				fpr++;
			else
				if (labelsTest[i] != prediction && prediction == 0)
					fnr++;
				else
					tnr++;
	}
	ofstream myfile("results.txt");
	if (myfile.is_open()) {
		myfile << ("tpr:");
		myfile << (" %d.", tpr);
		myfile << ("\n");
		myfile << ("fpr:");
		myfile << (" %d.", fpr);
		myfile << ("\n");
		myfile << ("fnr:");
		myfile << (" %d.", fnr);
		myfile << ("\n");
		myfile << ("tnr:");
		myfile << (" %d.", tnr);
	}
	myfile.close();
}

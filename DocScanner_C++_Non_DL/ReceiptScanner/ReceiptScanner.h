#pragma once
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <string>

using std::string;
using std::vector;
using std::pair;

typedef cv::Point Point;
typedef cv::Mat Matrix;
typedef vector<Matrix> VMat;
typedef vector<Point> VPoint;
typedef pair<int, int> PInt;

class ReceiptScanner
{
public:
	ReceiptScanner();
	~ReceiptScanner();

	VMat DetectReceipt(const vector<string>& imagePath);

private:
	VPoint OrderPoints(const VPoint _points);
	void MaskBackGround(const VPoint _points, Matrix& img)const;
	void Show(const Matrix& _img, const string& _name, const Matrix& _maskedImg, const string& _nameMasked)const;
	
};


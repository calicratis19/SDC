#include "ReceiptScanner.h"
#include "Utilities.h"
#include <iostream>
#include <algorithm>

using std::cout;
using std::endl;
using std::max;
using std::min;

ReceiptScanner::ReceiptScanner(){}

ReceiptScanner::~ReceiptScanner(){}

VMat ReceiptScanner::DetectReceipt(const vector<string>& imagePath)
{
	Matrix imgOriginal;        // input image
	Matrix imgGrayscale;       // grayscale of input image
	Matrix imgBlurred;         // intermediate blured image
	Matrix imgCanny;           // Canny edge image
	Matrix imgCanny32;           // Canny edge image

	vector<VPoint> contours;
	VPoint _points;
	cv::Scalar _color(255, 0, 0);
	VMat _imageList;


	for (const string& _path : imagePath)
	{
		imgOriginal = cv::imread(_path);          // open image
		if (imgOriginal.empty())				 // if unable to open image
		{         
			cout << "Could not read image: " << _path << endl;
			continue;
		}
		cv::cvtColor(imgOriginal, imgGrayscale, CV_BGR2GRAY);       // convert to grayscale


		cv::GaussianBlur(imgGrayscale,          // input image
			imgBlurred,                         // output image
			cv::Size(5, 5),                     // smoothing window width and height in pixels
			0);                                 

		cv::Canny(imgBlurred,           // input image
			imgCanny,                   // output image
			100,                        // low threshold
			200);                       // high threshold

		imgCanny.convertTo(imgCanny32, CV_32SC1,256);

		cv::findContours(imgCanny32, contours, CV_RETR_FLOODFILL, CV_CHAIN_APPROX_NONE);
		
		_points.clear();
		for (VPoint _vp : contours)
		{
			for (Point _point : _vp)
			{
				_points.push_back(_point);
				
			}
		}

		if (_points.size() < 4)
		{
			cout << " Could not detect receipt on image: " << _path << endl;
			continue;
		}

		_points = OrderPoints(_points);
		Matrix _maskedImage = imgOriginal.clone();

		cv::line(imgOriginal, _points[0], _points[1], _color, 10);
		cv::line(imgOriginal, _points[1], _points[2], _color, 10);
		cv::line(imgOriginal, _points[2], _points[3], _color, 10);
		cv::line(imgOriginal, _points[3], _points[0], _color, 10);

		MaskBackGround(_points, _maskedImage);
		_imageList.push_back(imgOriginal);

		cout << _points[0].x << " " << _points[0].y << endl;
		cout << _points[1].x << " " << _points[1].y << endl;
		cout << _points[2].x << " " << _points[2].y << endl;
		cout << _points[3].x << " " << _points[3].y << endl;

		Show(imgOriginal,"Final Output", _maskedImage, "Masked output");

	}	
	return _imageList;
}

VPoint ReceiptScanner::OrderPoints(const VPoint _points)
{
	VPoint _rect;
	

	PInt _sumMax;
	PInt _sumMin;
	PInt _diffMax;
	PInt _diffMin;

	for (int i=0;i<_points.size();i++)
	{
		Point _point = _points[i];

		int _sum = _point.x + _point.y;
		int _diff = _point.y - _point.x;

		if (!i)
		{
			_sumMin.first = _sumMax.first = _sum;
			_diffMin.first = _diffMax.first = _diff;
			_sumMin.second = _sumMax.second = _diffMin.second = _diffMax.second = i;
			continue;
		}
		
		if (_sumMax.first < _sum)
		{
			_sumMax.first = _sum;
			_sumMax.second = i;
		}
		if (_sumMin.first > _sum)
		{
			_sumMin.first = _sum;
			_sumMin.second = i;
		}
		if (_diffMax.first < _diff)
		{
			_diffMax.first = _diff;
			_diffMax.second = i;
		}
		if (_diffMin.first > _diff)
		{
			_diffMin.first = _diff;
			_diffMin.second = i;
		}
	}
	
	_rect.push_back(_points[_sumMin.second]);
	_rect.push_back(_points[_diffMax.second]);
	_rect.push_back(_points[_sumMax.second]);
	_rect.push_back(_points[_diffMin.second]);

	return _rect;
}

void ReceiptScanner::MaskBackGround(const VPoint _points, Matrix& img)const
{

	Util::Point _polygon[4];
	for (int k = 0; k < 4; k++)
	{
		_polygon[k] = { _points[k].x,img.rows - _points[k].y };
	}

	
	Util::Point _point;
	for (int i = 0;i < img.cols;i++)
	{
		for (int j = 0; j < img.rows;j++)
		{
			_point = Util::Point(j,img.cols-i);
			
			if (!Util::pnpoly(4,_polygon, _point))
			{
				img.at<cv::Vec3b>(i, j)[0] = 0;
				img.at<cv::Vec3b>(i, j)[1] = 0;
				img.at<cv::Vec3b>(i, j)[2] = 0;
			}
			
		}
	}
}

void ReceiptScanner::Show(const Matrix& _img, const string& _name, const Matrix& _maskedImg, const string& _nameMasked)const
{
	cv::imshow(_name, _img);
	cv::imshow(_nameMasked, _maskedImg);
	cv::waitKey(0);
	cv::destroyAllWindows();
}


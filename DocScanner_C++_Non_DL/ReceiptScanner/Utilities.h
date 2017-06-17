#pragma once
#include<vector>
using std::vector;
using std::pair;

namespace Util
{
	struct Point
	{
		int x;
		int y;
		Point(int _x, int _y) :x(_x), y(_y) {}
		Point(){}
	};


	int findNumberOfFilesInDirectory(std::string& path);	
	int pnpoly(int nvert, Point polygon[], Point p);
}

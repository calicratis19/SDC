#include <iostream>
#include "ReceiptScanner.h"
#include "Utilities.h"
using namespace std;
int main()
{
	ReceiptScanner _scanner;
	string _imagePrefix = "./images/";
	string _imagePostfix = ".jpg";
	vector<string>_nameList;

	int _numberOfImages = 150;	
	for (int i =0; i <_numberOfImages; i++)
	{
		_nameList.push_back(_imagePrefix+std::to_string(i+1)+_imagePostfix);		
	}	
	_scanner.DetectReceipt(_nameList);
	getchar();
	return 0;
}
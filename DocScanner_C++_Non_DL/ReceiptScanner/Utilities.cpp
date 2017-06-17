#include "Utilities.h"
#include <windows.h>
#define INF 10000

namespace Util
{
	int findNumberOfFilesInDirectory(std::string& path)
	{
		int counter = 0;
		WIN32_FIND_DATA ffd;
		HANDLE hFind = INVALID_HANDLE_VALUE;

		// Start iterating over the files in the path directory.
		hFind = ::FindFirstFileA(path.c_str(), &ffd);
		if (hFind != INVALID_HANDLE_VALUE)
		{
			do // Managed to locate and create an handle to that folder.
			{
				counter++;
			} while (::FindNextFile(hFind, &ffd) == TRUE);
			::FindClose(hFind);
		}
		else {
			printf("Failed to find path: %s", path.c_str());
		}

		return counter;
	}

	int pnpoly(int nvert, Point polygon[], Point p)
	{
		int i, j, c = 0;
		for (i = 0, j = nvert - 1; i < nvert; j = i++) {
			if (((polygon[i].y>p.y) != (polygon[j].y>p.y)) &&
				(p.x < (polygon[j].x - polygon[i].x) * (p.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x))
				c = !c;
		}
		return c;
	}
}
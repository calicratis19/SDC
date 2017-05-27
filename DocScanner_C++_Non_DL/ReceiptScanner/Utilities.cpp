#include "Utilities.h"
#include <windows.h>


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

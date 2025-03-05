#pragma once
#include "sqlite3.h"
#include <string>

class DBManager
{
public:
	DBManager();
	bool open();

	void close();
	bool insertMessage(const std::string& conversation,
		const std::string& sender,
		const std::string& message,
		const std::string& timestamp);
	static int chatHistoryCallback(void* data, int argc, char** argv, char** azColName);
	bool getChatHistory(const std::string& conversation,
		void* data);

private:

	template<typename T>
	static int callbackGenericNum(void* data, int argc, char** argv, char** azColName)
	{
		T* result = (T*)data;
		if (argv[0] == NULL)
		{
			*result = 0;
			return 0;
		}

		*result = (atof(argv[0]));

		return 0;
	}


	sqlite3* db;

};


#pragma once

#include <WinSock2.h>
#include <Windows.h>
#include <thread>
#include <queue>
#include <string>
#include "Helper.h"
#include <condition_variable>
#include <mutex>
#include "DBManager.h"
#include <map>
#include <vector>

class Server
{
public:
	Server();
	~Server();
	void serve(int port);

private:
	void acceptClient();
	void clientHandler(SOCKET clientSocket);
	void referClientToOpenThread();
	void writeMessagesToUsersFile();
	void writeMsgsToDB();
	std::string getAllActiveUsersStr();
	


	SOCKET _serverSocket;
	std::queue<std::pair<std::string, std::vector<std::string/*which users worte the message*/>>> _messages;
	std::map<std::string, SOCKET> _users;
	std::condition_variable _condNewMessage;

	std::mutex mtx;
	DBManager dbManager;

	

};


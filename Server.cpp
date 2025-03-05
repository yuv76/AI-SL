#include "Server.h"
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <iostream>
#include <fstream>
#include "DBManager.h"

#define AUTHOR_INDEX 0
#define SECOND_USER_INDEX 1
Server::Server()
{

	// this server use TCP. that why SOCK_STREAM & IPPROTO_TCP
	// if the server use UDP we will use: SOCK_DGRAM & IPPROTO_UDP
	_serverSocket = socket(AF_INET,  SOCK_STREAM,  IPPROTO_TCP); 

	if (_serverSocket == INVALID_SOCKET)
		throw std::exception(__FUNCTION__ " - socket");
}

Server::~Server()
{
	try
	{
		// the only use of the destructor should be for freeing 
		// resources that was allocated in the constructor
		closesocket(_serverSocket);
	}
	catch (...) {}
}

void Server::serve(int port)
{
	
	struct sockaddr_in sa = { 0 };
	
	sa.sin_port = htons(port); // port that server will listen for
	sa.sin_family = AF_INET;   // must be AF_INET
	sa.sin_addr.s_addr = INADDR_ANY;    // when there are few ip's for the machine. We will use always "INADDR_ANY"

	// Connects between the socket and the configuration (port and etc..)
	if (bind(_serverSocket, (struct sockaddr*)&sa, sizeof(sa)) == SOCKET_ERROR)
		throw std::exception(__FUNCTION__ " - bind");
	
	// Start listening for incoming requests of clients
	if (listen(_serverSocket, SOMAXCONN) == SOCKET_ERROR)
		throw std::exception(__FUNCTION__ " - listen");
	std::cout << "Listening on port " << port << std::endl;

	std::thread referClient(&Server::referClientToOpenThread, std::ref(*this));
	//writeMessagesToUsersFile();
	writeMsgsToDB();
	referClient.join();
	
	
	
}






void Server::acceptClient()
{

	// this accepts the client and create a specific socket from server to this client
	// the process will not continue until a client connects to the server
	SOCKET client_socket = accept(_serverSocket, NULL, NULL);

	if (client_socket == INVALID_SOCKET)
		throw std::exception(__FUNCTION__);

	std::cout << "Client accepted. creating a new thread" << std::endl;
	// the function that handle the conversation with the client
	std::thread thread(&Server::clientHandler, std::ref(*this),client_socket);
	/*clientHandler(client_socket);*/
	thread.detach();
}


void Server::clientHandler(SOCKET clientSocket)
{
	MessageType codeType = MT_CLIENT_LOG_IN;
	std::string username = "";
	try 
	{
		// get code type
		codeType = static_cast<MessageType>(Helper::getMessageTypeCode(clientSocket));

		// get the username
		username = Helper::getStringPartFromSocket(clientSocket, Helper::getIntPartFromSocket(clientSocket, 2));
	}
	catch (const std::exception& e) { closesocket(clientSocket); return; }
	try
	{
		
		std::string secondUsername = "";
		std::string data = "";
		std::unique_lock<std::mutex> locker(mtx,std::defer_lock);
		SOCKET sendToSocket = clientSocket;


		// if log in
		if (codeType == MT_CLIENT_LOG_IN)
		{
			// check that client isn't in the list yet
			if (!_users.count(username))
			{
				std::cout << "Welcome: " + username << std::endl;
				// add  to the set
				_users.insert(std::pair<std::string, SOCKET>(username, clientSocket));
			}
			

			//  send log in response
			Helper::send_update_message_to_client(sendToSocket, data, secondUsername, getAllActiveUsersStr());

			while (codeType != MT_CLIENT_FINISH && codeType != MT_CLIENT_EXIT && codeType != MT_CLIENT_DISCONNECTED)
			{
				
				// wait until a new message got written
				codeType = static_cast<MessageType>(Helper::getMessageTypeCode(clientSocket));


				if (codeType == MT_CLIENT_UPDATE)
				{
					sendToSocket = clientSocket;
					// get who the message is being sent to 
					secondUsername = Helper::getStringPartFromSocket(clientSocket, Helper::getIntPartFromSocket(clientSocket, 2));
					// get the message
					data = Helper::getStringPartFromSocket(clientSocket, Helper::getIntPartFromSocket(clientSocket, 5));

					
					if (!data.empty())
					{
						std::cout << "The message: " + data + ", is being sent to: " + secondUsername << std::endl;
					}

					locker.lock();
					/*** critical section start  ***/
					_messages.push(std::pair<std::string, std::vector<std::string>>(data, std::vector<std::string>{username, secondUsername}));
					/*** critical section end  ***/

					locker.unlock();


					// notify the write function that the queue had been updated
					_condNewMessage.notify_one();

				}
			}
			// remove user from active users set
			_users.erase(username);

		}
		
		
		closesocket(clientSocket); 
	}
	catch (const std::exception& e)
	{
		if (_users.find(username) != _users.end())
		{
			_users.erase(username);
		}
		closesocket(clientSocket);
	}


}

void Server::referClientToOpenThread()
{
	while (true)
	{
		// the main thread is only accepting clients 
		// and add then to the list of handlers
		std::cout << "Waiting for client connection request" << std::endl;
		acceptClient();
	}
}

// function writes messages from the messages queue and sends a response from the server to the clients accroding to the queue
void Server::writeMessagesToUsersFile()
{
	std::fstream outputFile;
	std::vector<std::string> usernames;
	std::string username, secondUsername;
	std::string currMsg = "";
	std::string data = "";
	
	
	while (true)
	{
		data = "";
		std::unique_lock<std::mutex> locker(mtx);
		// wait until there is a new message
		_condNewMessage.wait(locker, [&]() {return !_messages.empty(); }); // spurious awake


		/*** critical section start  ***/
		usernames = _messages.front().second;
		currMsg = _messages.front().first;
		_messages.pop();

		/*** critical section end  ***/
		locker.unlock();

		username = usernames[AUTHOR_INDEX];
		secondUsername = usernames[SECOND_USER_INDEX];
		if (secondUsername!= "")
		{
			// sort usernames
			std::sort(usernames.begin(), usernames.end());

			if (currMsg != "")
			{
				// start the message that will be written to the file
				//  (ref: &MAGSH_MESSAGE&&Author&<author_username>&DATA&<message_data>)
				currMsg = "&MAGSH_MESSAGE&&Author&" + username + "&DATA&" + currMsg;



				// create/open file with the username sorted as the name of it
				outputFile.open(usernames[0] + "&" + usernames[1] + ".txt", std::ios::out | std::ios::app);
				// add message to file
				outputFile << currMsg;
				outputFile.close();


			}

			// get all prev content of the file
			outputFile.open(usernames[0] + "&" + usernames[1] +".txt", std::ios::in);
			if (outputFile.is_open())
			{
				std::getline(outputFile, data);
				outputFile.close();
			}
		}


		Helper::send_update_message_to_client(_users[username], data, secondUsername, getAllActiveUsersStr());
	}
}

void Server::writeMsgsToDB()
{
	std::string data = "";
	std::vector<std::string> usernames;
	std::string username, secondUsername;
	std::string currMsg = "";

	while (true) {
		data = "";
		{
			// Lock and wait until there is a new message.
			std::unique_lock<std::mutex> locker(mtx);
			_condNewMessage.wait(locker, [&]() { return !_messages.empty(); });

			// Critical section: retrieve the next message.
			usernames = _messages.front().second;
			currMsg = _messages.front().first;
			_messages.pop();
		} // Unlock here.

		// Extract individual usernames (using your defined indices)
		username = usernames[AUTHOR_INDEX];
		secondUsername = usernames[SECOND_USER_INDEX];

		if (!secondUsername.empty()) {
			// Create a conversation identifier by sorting the usernames.
			std::sort(usernames.begin(), usernames.end());
			std::string conversationID = usernames[0] + "&" + usernames[1];

			if (!currMsg.empty()) {
				currMsg = "&MAGSH_MESSAGE&&Author&" + username + "&DATA&" + currMsg;

				// Get the current timestamp.
				std::string timestamp = Helper::getCurrentTimestamp();

				// Insert the message into the database.
				if (!dbManager.insertMessage(conversationID, username, currMsg, timestamp)) {
					std::cerr << "Failed to insert chat message into DB" << std::endl;
				}
			}

			// Retrieve chat history for the conversation.
			std::string chatHistory;
			if (dbManager.getChatHistory(conversationID, &chatHistory)) {
				data = chatHistory;
			}
			else {
				std::cerr << "Error retrieving chat history from DB" << std::endl;
			}
		}

		// Send updated chat history to the client.
		Helper::send_update_message_to_client(_users[username], data, secondUsername, getAllActiveUsersStr());
	}
}

std::string Server::getAllActiveUsersStr()
{
	std::string allUsersStr = "";
	for (const auto& user : _users) allUsersStr += user.first + '&';
	allUsersStr.pop_back();
	return allUsersStr;
}





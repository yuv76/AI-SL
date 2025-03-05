#include "DBManager.h"
#include <iostream>
#include "sqlite3.h"
#include <io.h>
#include <string>

#define DB_NAME "DB_MSG_HISTORY."

DBManager::DBManager()
{
    open();
}

bool DBManager::open()
{

    std::string dbFileName = DB_NAME;
    int file_exist = _access(dbFileName.c_str(), 0);
    int res = sqlite3_open(dbFileName.c_str(), &db);
    if (res != SQLITE_OK) {
        db = nullptr;
        std::cout << "Failed to open DB" << std::endl;
        return -1;
    }
    if (file_exist != 0) {
        // init database
        const char* sqlStatement = "CREATE TABLE IF NOT EXISTS ChatHistory ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "conversation TEXT NOT NULL, "
            "sender TEXT NOT NULL, "
            "message TEXT NOT NULL, "
            "timestamp TEXT NOT NULL);";

        char* errMessage = nullptr;
        res = sqlite3_exec(db, sqlStatement, nullptr, nullptr, &errMessage);
        if (res != SQLITE_OK)
        {
            std::cout << "ERROR CREATING DB! " << errMessage << std::endl;
            return false;
        }

        return true;

    }

    return true;

}

void DBManager::close()
{
    sqlite3_close(db);
    db = nullptr;
}

bool DBManager::insertMessage(const std::string& conversation,
    const std::string& sender,
    const std::string& message,
    const std::string& timestamp) {
    std::string sql = "INSERT INTO ChatHistory (conversation, sender, message, timestamp) VALUES ('"
        + conversation + "', '" + sender + "', '" + message + "', '" + timestamp + "');";
    char* errMessage = nullptr;
    int res = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errMessage);
    if (res != SQLITE_OK) {
        std::cerr << "ERROR: Couldn't insert message into DB! " << errMessage << std::endl;
        sqlite3_free(errMessage);
        return false;
    }
    return true;
}

int DBManager::chatHistoryCallback(void* data, int argc, char** argv, char** azColName)
{
    std::string* history = static_cast<std::string*>(data);
    if (argc >= 2) {
        *history += std::string(argv[0]) + ": " + argv[1]+"\n";
    }
    return 0;
}

bool DBManager::getChatHistory(const std::string &conversation,
                               void* data) {
    std::string sql = "SELECT sender, message FROM ChatHistory WHERE conversation = '" 
                      + conversation + "' ORDER BY timestamp;";
    char* errMessage = nullptr;
    int res = sqlite3_exec(db, sql.c_str(), chatHistoryCallback, data, &errMessage);
    if (res != SQLITE_OK) {
        std::cerr << "Error retrieving chat history: " << errMessage << std::endl;
        sqlite3_free(errMessage);
        return false;
    }
    return true;
}



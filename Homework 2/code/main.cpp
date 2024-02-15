#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

// Function to print the state of the board
void print_board(vector<vector<char>> &board)
{
    for (int i = 0; i < board.size(); i++)
    {
        vector<char> board_line = board[i];
        for (int j = 0; j < board_line.size(); j++)
        {
            cout << board_line[j] << "    ";
        }
        cout << "\n\n";
    }
    
}

int main()
{
    // Read the input file
    ifstream input_file("input.txt");

    // Read the player and opponent variables
    string player, opponent;
    getline(input_file, player);
    opponent = player == "X" ? "O" : "X";

    // Read the time remaining
    string time_remain;
    double player_time, opponent_time;
    getline(input_file, time_remain);
    istringstream iss(time_remain);
    iss >> player_time >> opponent_time;

    // Read the state of the board
    vector<vector<char>> board(12);
    for (int i = 0; i < 12; ++i)
    {
        string board_line_str;
        getline(input_file, board_line_str);
        vector<char> board_line(12);
        for (int j = 0; j < 12; j++)
        {
            board_line.push_back(board_line_str[j]);
        }
        board.push_back(board_line);
    }
    // print_board(board);

    return 0;
}

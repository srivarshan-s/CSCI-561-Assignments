#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

// Global constant to declare the board size
const int BOARD_SIZE = 12;

// Class to maintain game state
class GameState
{
private:
    vector<vector<char>> board;
    char player;
    char opponent;
    int value;
    bool player_turn;

public:
    // Constructor
    GameState(vector<vector<char>> board, char player, char opponent, bool turn)
    {
        this->board = board;
        this->player = player;
        this->opponent = opponent;
        this->value = this->evaluate();
        this->player_turn = turn;
    }

    // Function to print the state of the board
    void print_board()
    {
        for (int i = 0; i < BOARD_SIZE; i++)
        {
            for (int j = 0; j < BOARD_SIZE; j++)
            {
                cout << this->board[i][j] << "    ";
            }
            cout << "\n\n";
        }
    }

    // Function to evaluate the state of the board
    int evaluate()
    {
        int val;
        // Count the number of X and O
        int num_x = 0;
        int num_o = 0;
        for (int i = 0; i < BOARD_SIZE; i++)
        {
            for (int j = 0; j < BOARD_SIZE; j++)
            {
                if (this->board[i][j] == 'X')
                    num_x++;
                if (this->board[i][j] == 'O')
                    num_o++;
            }
        }
        // Since O always starts X get +1 bonus
        num_x++;
        // Value is difference between num of X and O
        if (player == 'X')
            val = num_x - num_o;
        else
            val = num_o - num_x;
        return val;
    }

    // Function to return the valid moves
    vector<pair<int, int>> valid_moves()
    {
        vector<pair<int, int>> moves;
        // Iterate through the board cells
        for (int i = 0; i < BOARD_SIZE; i++)
        {
            for (int j = 0; j < BOARD_SIZE; j++)
            {
                // Check if cell is empty
                if (board[i][j] == EMPTY)
                {
                    // Check if any one of the adjacent cells are filled
                    bool flag = false;
                    for (int x = i - 1; x <= i + 1; x++)
                    {
                        for (int y = j - 1; y <= j + 1; y++)
                        {
                            int x_idx, y_idx;
                            x_idx = max(0, min(x, BOARD_SIZE - 1));
                            y_idx = max(0, min(y, BOARD_SIZE - 1));
                            if (board[x_idx][y_idx] == WHITE || board[x_idx][y_idx] == BLACK)
                                flag = true;
                        }
                    }
                    if (flag)
                        moves.push_back(make_pair(i, j));
                }
            }
        }
        return moves;
    }
};

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
    vector<vector<char>> board;
    for (int i = 0; i < BOARD_SIZE; ++i)
    {
        string board_line_str;
        getline(input_file, board_line_str);
        vector<char> board_line;
        for (int j = 0; j < BOARD_SIZE; j++)
        {
            board_line.push_back(board_line_str[j]);
        }
        board.push_back(board_line);
    }

    // Initialize GameState object
    GameState start_state(board, player[0], opponent[0], true);
    start_state.print_board();

    return 0;
}

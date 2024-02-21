#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

// Global constant to declare the board size
const int BOARD_SIZE = 12;

// Global constants for board cells
const char WHITE = 'O';
const char BLACK = 'X';
const char EMPTY = '.';

// Class to maintain game state
class GameState
{
private:
    vector<vector<char>> board;
    char player;
    char opponent;
    bool player_turn;

public:
    // Constructor
    GameState(vector<vector<char>> board, char player, char opponent, bool turn)
    {
        this->board = board;
        this->player = player;
        this->opponent = opponent;
        this->player_turn = turn;
    }

    // Function to return player_turn
    bool turn() {
        return this->player_turn;
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
        // Count the number of white and black
        int num_white = 0;
        int num_black = 0;
        for (int i = 0; i < BOARD_SIZE; i++)
        {
            for (int j = 0; j < BOARD_SIZE; j++)
            {
                if (this->board[i][j] == WHITE)
                    num_white++;
                if (this->board[i][j] == BLACK)
                    num_black++;
            }
        }
        // Since white always starts black get +1 bonus
        num_black++;
        // Value is difference between num of white and black
        if (player == WHITE)
            val = num_white - num_black;
        else
            val = num_black - num_white;
        return val;
    }

    // Function to return the valid moves
    vector<pair<int, int>> valid_moves()
    {
        // Initialize vector to store moves
        vector<pair<int, int>> moves;
        // Choose white or black to check acccording to player turn
        char cell_to_check, cell_to_flip;
        if (player_turn)
        {
            cell_to_check = this->opponent;
            cell_to_flip = this->player;
        }
        else
        {
            cell_to_check = this->player;
            cell_to_flip = this->opponent;
        }
        // Iterate through the board cells
        for (int i = 0; i < BOARD_SIZE; i++)
        {
            for (int j = 0; j < BOARD_SIZE; j++)
            {
                // Check if cell is empty
                if (board[i][j] == EMPTY)
                {
                    bool flag = false;
                    int x, y;

                    // Check left of empty cell
                    if (j - 1 > 0 && board[i][j - 1] == cell_to_check)
                    {
                        // Check left cells till cell_to_flip is reached
                        for (y = j - 1; y >= 0; y--)
                        {
                            if (board[i][y] == EMPTY)
                                break;
                            if (board[i][y] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check right of empty cell
                    if (j + 1 < BOARD_SIZE - 1 && board[i][j + 1] == cell_to_check)
                    {
                        // Check right cells till cell_to_flip is reached
                        for (y = j + 1; y < BOARD_SIZE; y++)
                        {
                            if (board[i][y] == EMPTY)
                                break;
                            if (board[i][y] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check up of empty cell
                    if (i - 1 > 0 && board[i - 1][j] == cell_to_check)
                    {
                        // Check upper cells till cell_to_flip is reached
                        for (x = i - 1; x >= 0; x--)
                        {
                            if (board[x][j] == EMPTY)
                                break;
                            if (board[x][j] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check down of empty cell
                    if (i + 1 < BOARD_SIZE - 1 && board[i + 1][j] == cell_to_check)
                    {
                        // Check lower cells till cell_to_flip is reached
                        for (x = i + 1; x < BOARD_SIZE; x++)
                        {
                            if (board[x][j] == EMPTY)
                                break;
                            if (board[x][j] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check top left of empty cell
                    if (i - 1 > 0 && j - 1 > 0 && board[i - 1][j - 1] == cell_to_check)
                    {
                        // Check further cells till cell_to_flip is reached
                        for (x = i - 1, y = j - 1; x >= 0 && y >= 0; x--, y--)
                        {
                            if (board[x][y] == EMPTY)
                                break;
                            if (board[x][y] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check top right of empty cell
                    if (i - 1 > 0 && j + 1 < BOARD_SIZE - 1 && board[i - 1][j + 1] == cell_to_check)
                    {
                        // Check further cells till cell_to_flip is reached
                        for (x = i - 1, y = j + 1; x >= 0 && y < BOARD_SIZE; x--, y++)
                        {
                            if (board[x][y] == EMPTY)
                                break;
                            if (board[x][y] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check bottom left of empty cell
                    if (i + 1 < BOARD_SIZE - 1 && j - 1 > 0 && board[i + 1][j - 1] == cell_to_check)
                    {
                        // Check further cells till cell_to_flip is reached
                        for (x = i + 1, y = j - 1; x < BOARD_SIZE && y >= 0; x++, y--)
                        {
                            if (board[x][y] == EMPTY)
                                break;
                            if (board[x][y] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }

                    // Check bottom right of empty cell
                    if (i + 1 < BOARD_SIZE - 1 && j + 1 < BOARD_SIZE - 1 && board[i + 1][j + 1] == cell_to_check)
                    {
                        // Check further cells till cell_to_flip is reached
                        for (x = i + 1, y = j + 1; x < BOARD_SIZE && y < BOARD_SIZE; x++, y++)
                        {
                            if (board[x][y] == EMPTY)
                                break;
                            if (board[x][y] == cell_to_flip)
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                    {
                        moves.push_back(make_pair(i, j));
                        continue;
                    }
                }
            }
        }
        return moves;
    }

    // Function to apply a move
    GameState play(pair<int, int> move)
    {
        vector<vector<char>> new_board = this->board;

        char cell_to_flip = this->player_turn ? this->player : this->opponent;
        int x = move.first;
        int y = move.second;
        new_board[x][y] = cell_to_flip;

        // Check left of cell
        if (y - 1 > 0)
        {
            bool flag = false;
            int j;
            for (j = y - 2; j >= 0; j--)
            {
                if (new_board[x][j] == EMPTY)
                    break;
                if (new_board[x][j] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; j < y; j++)
                {
                    new_board[x][j] = cell_to_flip;
                }
            }
        }

        // Check right of cell
        if (y + 1 < BOARD_SIZE - 1)
        {
            bool flag = false;
            int j;
            for (j = y + 2; j < BOARD_SIZE; j++)
            {
                if (new_board[x][j] == EMPTY)
                    break;
                if (new_board[x][j] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; j > y; j--)
                {
                    new_board[x][j] = cell_to_flip;
                }
            }
        }

        // Check top of cell
        if (x - 1 > 0)
        {
            bool flag = false;
            int i;
            for (i = x - 2; i >= 0; i--)
            {
                if (new_board[i][y] == EMPTY)
                    break;
                if (new_board[i][y] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; i < x; i++)
                {
                    new_board[i][y] = cell_to_flip;
                }
            }
        }

        // Check bottom of cell
        if (x + 1 < BOARD_SIZE - 1)
        {
            bool flag = false;
            int i;
            for (i = x + 2; i < BOARD_SIZE; i++)
            {
                if (new_board[i][y] == EMPTY)
                    break;
                if (new_board[i][y] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; i > x; i--)
                {
                    new_board[i][y] = cell_to_flip;
                }
            }
        }

        // Check top left of cell
        if (x - 1 > 0 && y - 1 > 0)
        {
            bool flag = false;
            int i, j;
            for (i = x - 2, j = y - 2; i >= 0 && j >= 0; i--, j--)
            {
                if (new_board[i][j] == EMPTY)
                    break;
                if (new_board[i][j] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; i < x && j < y; i++, j++)
                {
                    new_board[i][j] = cell_to_flip;
                }
            }
        }

        // Check top right of cell
        if (x - 1 > 0 && y + 1 < BOARD_SIZE - 1)
        {
            bool flag = false;
            int i, j;
            for (i = x - 2, j = y + 2; i >= 0 && j < BOARD_SIZE; i--, j++)
            {
                if (new_board[i][j] == EMPTY)
                    break;
                if (new_board[i][j] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; i < x && j > y; i++, j--)
                {
                    new_board[i][j] = cell_to_flip;
                }
            }
        }

        // Check bottom left of cell
        if (x + 1 < BOARD_SIZE - 1 && y - 1 > 0)
        {
            bool flag = false;
            int i, j;
            for (i = x + 2, j = y - 2; i < BOARD_SIZE && j >= 0; i++, j--)
            {
                if (new_board[i][j] == EMPTY)
                    break;
                if (new_board[i][j] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; i > x && j < y; i--, j++)
                {
                    new_board[i][j] = cell_to_flip;
                }
            }
        }

        // Check bottom right of cell
        if (x + 1 < BOARD_SIZE - 1 && y + 1 < BOARD_SIZE)
        {
            bool flag = false;
            int i, j;
            for (i = x + 2, j = y + 2; i < BOARD_SIZE && j < BOARD_SIZE; i++, j++)
            {
                if (new_board[i][j] == EMPTY)
                    break;
                if (new_board[i][j] == cell_to_flip)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (; i > x && j > y; i--, j--)
                {
                    new_board[i][j] = cell_to_flip;
                }
            }
        }

        GameState new_state(new_board, this->player, this->opponent, !this->player_turn);
        return new_state;
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
    
    cout << "START STATE" << '\n';
    start_state.print_board();

    vector<pair<int, int>> moves = start_state.valid_moves();
    GameState prev_state = start_state;
    while (!moves.empty())
    {
        int max = -999;
        int min = 999;
        pair<int, int> next_move = moves[0];
        cout << "NEXT MOVE" << '\n';
        for (auto move: moves) {
            GameState temp_state = prev_state.play(move);
            int val = temp_state.evaluate();
            if (prev_state.turn() && val > max)
            {
                max = val;
                next_move = move;
            }
            if (!prev_state.turn() && val < min)
            {
                min = val;
                next_move = move;
            }
        }
        GameState next_state = prev_state.play(next_move);
        next_state.print_board();
        prev_state = next_state;
        moves = prev_state.valid_moves();
    }
    

    return 0;
}

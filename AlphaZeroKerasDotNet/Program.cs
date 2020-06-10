using ConnectFour;
using System;
using System.Collections.Generic;

namespace AlphaZeroKerasDotNet
{
    class Program
    {
        static User player1, player2;
        static void Main(string[] args)
        {
            bool firstRun = true;
            while (true)
            {
                play_one_round(firstRun);
                firstRun = false;
                Console.WriteLine("Do you want to play again (y/n)?");
                String play_again = Console.ReadLine();
                if (play_again.ToLower() != "y")
                {
                    Console.WriteLine("Good Bye..");
                    break;
                }
            }
        }

        static void play_one_round(bool firstRun)
        {
            Console.WriteLine("Do you want to go first (y/n)?");
            String go_first = Console.ReadLine();
            int isFirst = (go_first.ToLower() == "y") ? 1 : -1;
            int player1_version = -1;
            int player2_version = 206;
            int episodes = 1;

            if (firstRun)
                playMatchesBetweenVersions(player1_version, player2_version, episodes, 0, isFirst);
            else
                playMatches(player1, player2, episodes, 0, isFirst);
        }

        static void playMatchesBetweenVersions(int player1version, int player2version, int EPISODES, int turns_until_tau0, int goes_first = 0)
        {
            Residual_CNN player1_NN, player2_NN;

            if (player1version == -1)
                player1 = new User("player1", Game.StateSize, Game.ActionSize);
            else
            {
                player1_NN = new Residual_CNN(Config.REG_CONST, Config.LEARNING_RATE, Game.InputShape, Game.ActionSize);
                player1 = new Agent("player1", Game.StateSize, Game.ActionSize, Config.MCTS_SIMS, Config.CPUCT, player1_NN);
                if (player1version > 0)
                    player1_NN.ReadModel(player1version);
            }


            if (player2version == -1)
                player2 = new User("player2", Game.StateSize, Game.ActionSize);
            else
            {
                player2_NN = new Residual_CNN(Config.REG_CONST, Config.LEARNING_RATE, Game.InputShape, Game.ActionSize);
                player2 = new Agent("player2", Game.StateSize, Game.ActionSize, Config.MCTS_SIMS, Config.CPUCT, player2_NN);
                if (player2version > 0)
                    player2_NN.ReadModel(player2version);
            }

            playMatches(player1, player2, EPISODES, turns_until_tau0, goes_first);
        }

        static void playMatches(User player1, User player2, int EPISODES, int turns_until_tau0, int goes_first = 0)
        {
            Game env = new Game();
            GameState state = env.GameState;
            int done = 0;
            int turn = 0;
            int value = 0;
            player1.mcts = null;
            player2.mcts = null;
            int player1Starts = -1;
            for (int e = 0; e <= EPISODES - 1; e++)
            {
                state = env.Reset();
                player1Starts = (goes_first == 0) ? -player1Starts : goes_first;
                Dictionary<int, User> players = new Dictionary<int, User>();
                if (player1Starts == 1)
                {
                    players.Add(1, player1);
                    players.Add(-1, player2);
                    Console.WriteLine(player1.name + " plays as X");
                }
                else
                {
                    players.Add(1, player2);
                    players.Add(-1, player1);
                    Console.WriteLine(player2.name + " plays as X");
                }
                Console.WriteLine(state.ToString());

                //Start single match
                while (done == 0)
                {
                    turn += 1;
                    int tau = (turn < turns_until_tau0) ? 1 : 0;

                    //Run the MCTS algo and return an action
                    int action = players[state.PlayerTurn].Act(state, tau);

                    //Do the action
                    env.Step(action);

                    state = env.GameState;
                    value = env.GameState.Value;
                    done = Convert.ToInt32(env.GameState.Done);

                    Console.WriteLine(state.ToString());
                    if (done == 1)
                    {
                        if (value != 0)
                        {
                            Console.WriteLine("{0}s WINS!", players[-state.PlayerTurn].name);
                        }
                        else
                        {
                            Console.WriteLine("DRAW...");
                        }
                    }
                }
            }


        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;

namespace AlphaZeroKerasDotNet
{
    public static class Config
    {
        // SELF PLAY
        public static int EPISODES = 30;
        public static int MCTS_SIMS = 50;
        public static int MEMORY_SIZE = 3000;
        public static int TURNS_UNTIL_TAU0 = 10;

        public static int CPUCT = 1;
        public static float EPSILON = 0.2f;
        public static float ALPHA = 0.8f;



        // RETRAINING
        public static int BATCH_SIZE = 256;
        public static int EPOCHS = 1;
        public static float REG_CONST = 0.0001f;
        public static float LEARNING_RATE = 0.1f;
        public static float MOMENTUM = 0.9f;
        public static int TRAINING_LOOPS = 10;

    }
}

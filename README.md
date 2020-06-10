# AlphaZero methodology with C# and Keras .NET

The AlphaZero methodology is adapted from: https://adsp.ai/articles/how-to-build-your-own-alphazero-ai-using-python-and-keras/

The project is created using .NET Core console with Keras .NET imported as nuget package. Keras .NET is a high level neural network API written in C# with python binding, and enables for the model trained in python to be consumed by .NET code, thereby opening up the possibility to integrate machine-learning in .NET projects efficiently.  

The Connect Four game is trained using python and the model is imported for predicting moves against the human player. The neural network predictions are integrated with Monte Carlo Tree Search in a way that provides the MCTS with a kind of human intuition, as to overcome the shortcomings of standard MCTS.

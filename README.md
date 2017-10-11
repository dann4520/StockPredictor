Neural Network Stock Performance Predictor
Daniel Wood
dlwood5@cougars.ccis.edu

Overview
I plan to create a Neural Network that can predict the future returns of securities
based on past data.  This will be validated using backtesting on past data not used
in the training of the Neural Network.

Background
Efficient Market Hypothesis tells us that we cannot predict future security prices
because all currently available information is already priced into the security.
Only new information can affect future security prices.  I hope to prove this wrong
by creating a Neural Network which can successfully predict the future price of a
security within a reasonable level of accuracy.

Technical Analysis
Technical Analysis focuses on price and volume, ignores fundamentals such as company
earnings or company balance sheet. Looks for trends in past trade price and volume to
make predictions about the future price. A Neural Network will be fed past price and
volume data to identify and “learn” trends which can then be used to predict future
price behavior.
Key Indicators to be studied
Daily Adjusted Price
Daily Trade Volume
RSI (Derived from DAP)

RSI is the Relative Strength Index value calculated as RSI = 100 - 100 / (1+RS) where
RS = Average Gain of Period / Average Loss of Periodwhere a Period will be 14 days.
RSI is used to capture and smooth out trends.

The Neural Network will be written in Python 2.  I plan on using the following modules.
Tkinter to create a basic GUI (Time Permitting)
csv to handle csv files exported from Yahoo Finance.

Will need the NumPy Library for handling matrices.

Testing the Neural Network Prediction Accuracy
Backtesting will be used to test our model on historical data. Historical data used
for teaching Neural Network will be prior to January 1st 2016 so that all data from
January 1st 2016 to present could be used to test the accuracy of the model. I will
use the trained Neural Network to make monthly predictions starting with January 1st
2016 through November 30th 2017.  For example, the first test period will go from
January 1st 2016 through January 31st 2016.  A prediction for the period will be made
and then compared against the actual returns from the period.

Success will be gauged by how close my predictions are to the actual results.
An excellent result would be that I can predict a price change of a given period accurate
within 3% points of the actual return.  An acceptable result would be that I can predict
the correct direction, positive or negative return, at least 75% of the time.

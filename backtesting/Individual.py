import sys
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid
from iexfinance.stocks import get_historical_data
from datetime import datetime
from pyti.exponential_moving_average import exponential_moving_average as ema
from pyti.simple_moving_average import simple_moving_average as sma
from pyti.price_oscillator import price_oscillator as po

class Individual:
	def __init__(self, stock, shortwindow, longwindow, startdate, enddate, asset, strategy, buying_power):
		self.stock = stock
		self.short_window = shortwindow
		self.long_window = longwindow
		self.trading_rule = strategy
		self.asset = asset
		#self.name = str(startdate.year) + "-" + str(startdate.month) + "-" + str(startdate.day) + "_" + str(enddate.year) + "-" + str(enddate.month) + "-" + str(enddate.day) + "_" + stock
		self.name = str(uuid.uuid4())[0:10] + "_" + stock	
		self.start = startdate
		self.end = enddate
		self.initial_capital = float(100000.0)
		self.historical_returns = list()
		self.flag = False
		self.buy_quantity = buying_power

		self.benchmark = 0.0
		self.sharpe_ratio = 0.0
		self.alpha = 0.0
		self.absolute_return = 0.0

	def trading_strategy(self):
		if (self.strategy == 'SMA'):
			return 0

	def main(self):
		#account for multiple securities
		#document all quandl datasets
		#rate limited to 50 calls per day
		try:
			security = quandl.get(self.asset + "/" + self.stock, start_date=self.start, end_date=self.end)
		except:
			return 0

		security.head()
		security.to_csv('data/' + self.name + 'quandl' + '.csv')

		# price dimension conditional on database used 
		if (self.asset == 'WIKI'):
			dimension = 'Adj. Close'
			self.benchmark = quandl.get('CHRIS/CME_SP1', start_date=self.start, end_date=self.end)
			self.benchmark = self.benchmark['Settle']
		elif (self.asset == 'CHRIS' or 'EUREX'):
			dimension = 'Settle'
			self.benchmark = security[dimension]

		daily_close = security[[dimension]]
		data = daily_close.values.T.tolist()[0]

		'''

		self.price_oscillator = po(data, self.short_window, self.long_window)
		print(self.price_oscillator)

		emaplot = ema(data, 10)
		smaplot = sma(data, 10)
		plt.plot(emaplot[:])
		plt.plot(smaplot[:])
		plt.plot(data[:])
		#plt.show()
		'''

		if daily_close.empty:
			self.flag = True
			print("Dataframe is empty ", self.stock)
			return None

		# Show the plot
		daily_close.plot()
		plt.title(self.name)

		#trading signal generation

		signals = pd.DataFrame(index=security.index)
		signals['signal'] = 0.0

		# Create short simple moving average over the short window
		signals['short_mavg'] = security[dimension].rolling(window=self.short_window, min_periods=1, center=False).mean()
		#rolling window calculation of short_window, min_periods, center and then take the average of that value
		# Create long simple moving average over the long window
		# 'Close' dimension only applicable to equities
		signals['long_mavg'] = security[dimension].rolling(window=self.long_window, min_periods=1, center=False).mean()

		# Create signals
		signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:] 
		                                            > signals['long_mavg'][self.short_window:], 1.0, 0.0)   

		# Generate trading orders
		signals['positions'] = signals['signal'].diff()
		#signals dataFrame contains position dimension which is either 0 for no signal or 1 for buy and -1 for sell
		#if short is greater than long signals is 1, -1 for inverse

		# Initialize the plot figure
		fig = plt.figure()

		# Add a subplot and label for y-axis
		ax1 = fig.add_subplot(111,  ylabel='Price in $')

		# Plot the closing price
		security[dimension].plot(ax=ax1, color='r', lw=2.)

		# Plot the short and long moving averages
		signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

		# Plot the buy signals
		ax1.plot(signals.loc[signals.positions == 1.0].index, 
		         signals.short_mavg[signals.positions == 1.0],
		         '^', markersize=10, color='m')
		         
		# Plot the sell signals
		ax1.plot(signals.loc[signals.positions == -1.0].index, 
		         signals.short_mavg[signals.positions == -1.0],
		         'v', markersize=10, color='k')

		# Create a DataFrame `positions`
		positions = pd.DataFrame(index=signals.index).fillna(0.0)

		# Buy a 100 shares
		positions['security'] = 100*signals['signal']   
		  
		# Initialize the portfolio with value owned   
		portfolio = positions.multiply(security[dimension], axis=0)

		# Store the difference in shares owned 
		pos_diff = positions.diff()

		# Add `holdings` to portfolio
		portfolio['holdings'] = (positions.multiply(security[dimension], axis=0)).sum(axis=1)

		# Add `cash` to portfolio
		portfolio['cash'] = self.initial_capital - (pos_diff.multiply(security[dimension], axis=0)).sum(axis=1).cumsum()   

		# Add `total` to portfolio
		portfolio['total'] = portfolio['cash'] + portfolio['holdings']

		# Add `returns` to portfolio
		portfolio['returns'] = portfolio['total'].pct_change()

		fig = plt.figure()

		ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')

		# Plot the equity curve in dollars
		portfolio['total'].plot(ax=ax1, color='aqua', lw=2.)

		# Plot the "buy" trades against the equity curve
		ax1.plot(portfolio.loc[signals.positions == 1.0].index, 
		         portfolio.total[signals.positions == 1.0],
		         '^', markersize=10, color='m')

		# Plot the "sell" trades against the equity curve
		ax1.plot(portfolio.loc[signals.positions == -1.0].index, 
		         portfolio.total[signals.positions == -1.0],
		         'v', markersize=10, color='k')

		# Show the plot
		#plt.show()
		# Isolate the returns of your strategy
		returns = portfolio['returns']

		self.initial_capital = portfolio['total'][len(portfolio['total'])-1] #hardcoded should be changed
		# output all dataframes as individual csv files
		signals.to_csv('data/' + self.name + 'signals.csv')
		positions.to_csv('data/' + self.name + 'positions.csv')
		portfolio.to_csv('data/' + self.name + 'portfolio.csv')

		#prevent memory overload
		plt.close('all')

		# computing benchmark
		self.benchmark = ((self.benchmark[len(self.benchmark)-1] - self.benchmark[0]) / self.benchmark[0]) * 100
		#calculate returns for benchmark rather than average value
		#alternative formula for sharpe ratio
		self.sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
		#return over many generations
		self.absolute_return = ((self.initial_capital - 100000.00) / 100000.00) * 100
		#return over current generation
		self.average_return = returns.mean()
		#storing historical returns per generation
		self.historical_returns.append(self.absolute_return)
		self.historical_average_return = 0.0
		for i in self.historical_returns:
			self.historical_average_return += i
		self.historical_average_return = self.historical_average_return /  len(self.historical_returns)
		#return correlated with benchmark return
		self.alpha = self.absolute_return - self.benchmark
		self.alt_alpha = self.average_return - self.benchmark
# tensorflow-stock-prediction

Using tensorflow to predict stocks using an artifical neuronal net

## Output Format
	Instead of training on the last day's (scaled) price we target the rise and fall property directly
	The output will be (softscaled) 1 or 0, representing rise or fall

## Input Format
	We don't need the concrete values of our daily stockprices, as we are only interested in the percentual gains and losses.
	Instead of scaling the whole dataset at once, we now scale every dataset entry individually.
	Example: An input of [10,200,100,..]
		Scaled over the whole dataset could be [0.005,0.1,0.05,..]
	While: An input of [100,2000,1000]
		Scaled over the same dataset could be [0.4,0.95,0.7,..]

	Both inputs are semantically similar to us and scaling them per entry will make them also numercally more similar.
	If scaled per entry, the values will be more separated in a 0,1 interval for each entry [0.05,0.5,1,..]

## The new prepared data for our net
1.0000,0.3474,0.0000,0.3201,0.3855,0.4418,0.3517,0.8094
0.7863,0.0000,0.7246,0.8727,1.0000,0.7962,0.9936,0.1704
0.0000,0.7246,0.8727,1.0000,0.7962,0.9936,0.5321,0.8117
...

Every input sequence ranges from 0 to 1
Every output value ranges
	from 0 to 0.25 for a fall
	from 0.75 to 1 for a rise

## Results

### sigmoid
adadelta
MSE Train:  0.049233235
MSE Test:  0.04046173
Hitrate: 49.41%
grad desc
MSE Train:  0.090486966
MSE Test:  0.10966104
Hitrate: 50.00%
adam
MSE Train:  0.0066285883
MSE Test:  0.008100081
Hitrate: 48.78%
### relu
adadelta
MSE Train:  0.01285096
MSE Test:  0.013012319
Hitrate: 51.96%
grad
MSE Train:  0.0684103
MSE Test:  0.097318664
Hitrate: 51.61%
adam
MSE Train:  0.009341125
MSE Test:  0.011232391
Hitrate: 51.79%

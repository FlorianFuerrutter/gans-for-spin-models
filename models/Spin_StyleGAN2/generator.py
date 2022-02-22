
latent
FC
FC
FC
out W latent with 4 invidual vectors to inject -> A


const 8x8
A+WeightDemodulationConv2D
bias + B
------------------------------->ToRGB conv1x1 filter 3 + UP

bilin UP 16x16
A+WeightDemodulationConv2D
bias + B
A+WeightDemodulationConv2D
bias + B
------------------------------->ToRGB conv1x1 filter 3 + UP + ADD


bilin UP 32x32
A+WeightDemodulationConv2D
bias + B
A+WeightDemodulationConv2D
bias + B
------------------------------->ToRGB conv1x1 filter 3 + UP + ADD

bilin UP 64x64
A+WeightDemodulationConv2D
bias + B
A+WeightDemodulationConv2D
bias + B
------------------------------->ToRGB conv1x1 filter 3 + UP  + ADD --> OUTPUT!!!!!!!
# dependents
    Unity>=2018.2
    TensorflowSharp>=1.9.0  (1.12.0 is faster)
    jsonDotNat (included)
# install

### install TensorflowSharp
- [download](https://www.nuget.org/packages/TensorFlowSharp/) and rename .nuget to .zip
- unzip to Assets/Plugins/Tensorflow
- delete Tensorflow/lib/net471/* or Tensorflow/lib/netstandard2/* depends on your config

*note 1: I only tested on windows.

*note 2: It seems TensorflowSharp only works on cpu version.

# usage
- convert your model to the untiy interface .pb file
- cpoy the .pb file and labels map .json to Assets
- rename .pb to .bytes
# references
[TensorFlowSharp](https://github.com/migueldeicaza/TensorFlowSharp)

[Syn-McJ/TFClassify-Unity](https://github.com/Syn-McJ/TFClassify-Unity)
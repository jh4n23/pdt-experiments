type Image = Tensor Real [28, 28]

type Label = Index 10

@parameter
low : Real

@parameter
high : Real

-- check image pixels are between lower and upper bounds
-- pixel values may be normalised, so this is not necessarily as simple as low = 0 and high = 1
validImage : Image -> Bool
validImage x = forall i j . low <= x ! i ! j <= high

@network
classifier : Image -> Tensor Real [10]

advises : Image -> Label -> Bool
advises x i = forall j . classifier x ! i >= classifier x ! j

@parameter
epsilon : Real

boundedByEpsilon : Image -> Bool
boundedByEpsilon x = forall i j . -epsilon <= x ! i ! j <= epsilon

robustAround : Image -> Label -> Bool
robustAround image label = forall perturbation .
  let perturbedImage = image - perturbation in
  boundedByEpsilon perturbation and validImage perturbedImage =>
    advises perturbedImage label

scrAround : Image -> Label -> Bool
scrAround image label = forall perturbation .
    let perturbedImage = image - perturbation in
    boundedByEpsilon perturbation and validImage perturbedImage =>
        classifier perturbedImage ! label >= 0.52 -- arbitrary value of eta

@parameter(infer=True)
n : Nat

@dataset
trainingImages : Vector Image n

@dataset
trainingLabels : Vector Label n

@parameter
p : Real

qllAdditive : DifferentiableTensorLogic
qllAdditive =
  { trueElement                = -infinity
  , falseElement               = infinity
  , pointwiseNegation          = \x -> -x
  , pointwiseConjunction       = \{dims} x y -> (const (1/p) dims) * log(exp(const p dims * x) + exp(const p dims * y))
  , pointwiseDisjunction       = \{dims} x y -> -(const (1/p) dims) * log(exp(const (-p) dims * x) + exp(const (-p) dims * y))
  , pointwiseLessThan          = \x y -> x - y
  , pointwiseLessEqualThan     = \x y -> x - y
  , pointwiseGreaterThan       = \x y -> y - x
  , pointwiseGreaterEqualThan  = \x y -> y - x
  , pointwiseEqual             = \x y -> max (x - y) (y - x)
  , pointwiseNotEqual          = \x y -> - max (x - y) (y - x)
  , reduceConjunction          = \{dims} xs -> (1/p) * log(reduceAdd (exp (const p dims * xs)))
  , reduceDisjunction          = \{dims} xs -> (1/p) * log(reduceAdd (exp (const (-p) dims * xs)))
  }

@property
robust : Vector Bool n
robust = foreach i . robustAround (trainingImages ! i) (trainingLabels ! i)
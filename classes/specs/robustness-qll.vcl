--------------------------------------------------------------------------------
-- Inputs and outputs

-- Define the type for our input images
type Image = Tensor Real [1, 28, 28]

-- The type of the output labels
-- i.e a number between 0 and 9, one for each digit
type Label = Index 10

-- A predicate that states that all the pixel values in a given image are
-- in the range 0.0 to 1.0
validImage : Image -> Bool
validImage x = forall i j . 0 <= x ! 0 ! i ! j <= 1

--------------------------------------------------------------------------------
-- Network

-- Declare the network used to classify images. The output of the network is a
-- score for each of the digits 0 to 9.
@network
classifier : Image -> Tensor Real [1, 10]

-- The classifier advises that input image `x` has label `i` if the score
-- for label `i` is greater than or equal to the score of any other label `j`.
advises : Image -> Label -> Bool
advises x i = forall j . classifier x ! 0 ! i >= classifier x ! 0 ! j

--------------------------------------------------------------------------------
-- Definition of robustness around a point

-- First we define the parameter `epsilon` that will represent the radius of the
-- ball that we want the network to be robust in. Note that we declare this as
-- a parameter which allows the value of `epsilon` to be specified at compile
-- time rather than be fixed in the specification.
@parameter
epsilon : Real

-- Next we define what it means for an image `x` to be in a ball of
-- size epsilon around 0.
boundedByEpsilon : Image -> Bool
boundedByEpsilon x = forall i j . -epsilon <= x ! 0 ! i ! j <= epsilon

-- We now define what it means for the network to be robust around an image `x`
-- that should be classified as `y`. Namely, that for any perturbation no greater
-- than epsilon then if the perturbed image is still a valid image then the
-- network should still advise label `y` for the perturbed version of `x`.
robustAround : Image -> Label -> Bool
robustAround image label = forall perturbedImage .
  boundedByEpsilon (image - perturbedImage) and validImage perturbedImage =>
    advises perturbedImage label

--------------------------------------------------------------------------------
-- Robustness with respect to a single image in the dataset

@dataset
image : Image

@dataset
label : Vector Label 1

@property
robust : Bool
robust = robustAround image (label ! 0)

--------------------------------------------------------------------------------
-- Training to eliminate a perturbation that causes the network to misclassify an image

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

@dataset
perturbedImage : Image

@property
robust : Bool
robust = 
  -- Add a false dependency on the logic to avoid it being removed by monomorphisation.
  -- Will fix in the next version of Vehicle.
  if qllAdditive.trueElement > 0 
    then False 
    else advises perturbedImage (label ! 0)
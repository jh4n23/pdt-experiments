type Image = Tensor Real [28, 28]

type Label = Index 10

validImage : Image -> Bool
validImage x = forall i j . 0 <= x ! i ! j <= 1

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

@property
robust : Vector Bool n
robust = foreach i . robustAround (trainingImages ! i) (trainingLabels ! i)

@property
scr : Vector Bool n
scr = foreach i . scrAround (trainingImages ! i) (trainingLabels ! i)
# Learning From Positive and Unlabeled Data: A Survey 
Authors: Bekker & Davis<br>
Link: https://arxiv.org/pdf/1811.04820.pdf 

## Introduction

PU learning focuses on training an algorithm given a set of data with positive and unlabeled data. It is derived from positive-only (one-class) and semi-supervised learning.

PU learning is widely considered a special case of binary learning which is typically applicable in two settings: 
    1. Single Training Set
    2. Case Control Design

### Review of Binary Classification
- Goal: train a classifier that can distinguish between two classes of instances based on their attributes. 
- To enable training, the data is assumed to be an independent and identically distributed sample of the real distribution. 

### PU Learning
- The goal of PU learning is identical to binary classification EXCEPT during the learning phase, only some fo the (+) examples in the training data are labeled and none of the negative examples are.
- Labels are typically a set of triplets $(x,y,s)$ where $x$ is a vector of attributes, $y$ the class and $s$ a binary variable representing whether the tuple was selected to be a label. 

### Labeling Mechanism
- The labeled positive examples are selected from the complete set of (+) examples according to a probabilistic labeling mechanism, where each positive example $x$ has the probability $e(x) = Pr(s=1|y=1,x)$ of being selected to be labeled (this is the propensity score).
    - Thus the labeled distribution is a biased version of the positive distribution. 
    - This labeled instance space is dependent on the label frequency - $c$ (aka what fraction of positives retain their labels)

### Single Training Set Scenario
- Assumes that positive and unlabeled data examples come from the same dataset and that this dataset is an independent and identically distributed sample of the real distribution (just like supervised classification). A fraction of the data $c$ is sampled from the positive examples are selected to be labeled, following their individual propensity scores. 
- Label frequency $c$:
<br><center>$c = \frac{Pr(s=1)}{\alpha}$</center> 

- $Pr(s=1)$ can be counted in the data as the fraction of labeled examples
- $Pr(y=1)$ is related to the class prior. in the single-training-set-scenario, it is equal to the class prior. 

### Case-Control Scenario
- assumes that the positive and unlabeled examples come from two independent datasets tAND that the unlabeled dataset is an i.i.d. sample from the real distribution. 
- this is an example where you have positive examples - and the remaining data is randomly sampled from a selection of centers. 
- Label frequency $c$:
<br><center>$c = \frac{Pr(s=1)}{\alpha(1-Pr(s=1))+Pr(s=1)}$</center> 
<br><center>$\alpha = \frac{1-c}{c} * \frac{Pr(s=1)}{1-Pr(s=1)}$</center>

- $Pr(s=1)$ can be counted in the data as the fraction of labeled examples
- $Pr(y=1)$ is related to the class prior. in the case-control scenario, the class prior is defined in the unlabeled data.

- The observed positive examples are generated from the same distributions in both the single training-set and the case-control scenario. Thus in both cases **the learner has access to a set of examples drawn i.i.d. from the true distribution and a set of examples that are drawn from the true distribution and a set of examples that are drawn from the (+) distribution according to the labeling mechanism that is defined by the propensity score $e(x)$. 

## Key Assumptions
Two assumptions about why data is unlabeled: 
1. It is truly a negative example
2. It is a positive example, but simply was not selected by the labeling mechanism to have its label observed.

Thus it is necessary to make assumptions about the labeling mechanism, the class distributions in the data or both. 
    - to enable estimating class priors directly from the data -> more assumptions need to be made:

### Selected Completely At Random (SCAR)
Assumes that the set of labeled examples is a uniform subset of the set of positive examples. It is often motivated by the case-control study design, where it is often reasonable to assume the labeled dataset is an i.i.d. sample from the positive distribution. This enables a PU learning problem to be reduced to a standard binary classification task. 

Definition 1: Labeled examples are selected completely at random, independent from their attributes from the positive distribution. The propensity score $e(x)$ (probability for selecting a positive example is constant and equal to the label frequency)

Because the probability for an example to be labeled is directly proportional to the probability for an example to be positive, this enables **the use of non-traditional classifiers** which predict $Pr(s=1|x)$, which are learned by considering the unlabeled examples as negative. 
    - QUESTION: what the hell is a non-traditional classifier

Properties of non-traditional classifiers: 
    - non-traditional classifiers preserve the ranking order
    - training a traditional classifier subject to a desired expected recall (aka equivalent to training a non-traditional classifier subject to that recall)
    - given the label frequency, a probabilistic non-traditional classifier can be converted to a traditional classifier by dividing the outputs by the label frequency

Note: this is seperate to the missing completely at random assumption as the missingness must be class dependent. 

### Selected at Random (SAR)
- the probability for selecting positive examples to be labeled depends on its attribute values. 
- general assumption is motivated by the fact, that many PU learning applications suffer from labeling bias
    - Example: whether a patient suffering from a disease will visit a doctor depends on her socioeconomic status and severity of symptoms
- Labeled examples are a biased sample from the (+) distribution where the bias completely depends on the attribute and is defined by the propensity score. 
<br><center> $e(x) = Pr(s=1|x,y=1)$ </center>

### Probabilistic Gap PU (PGPU)
Labeled examples are a biased sample from the positive distribution, where examples with a smaller probabilistic gap $\Delta{Pr(x)}$ are less likely to be labeled.
Simply, those examples which are more "grey-zone" are less likely to be labeled. 

The propensity score is a non-negative, monotone increasing function f of the probabilistic gap 
<br><center> $e(x) = f(\Delta{Pr(x)}) = f(Pr(y-1|x) - Pr(y=0|x))$</center>

The observed probabilistic gap is related to the real probabilistic gap:
- The observed probabilistic gap is always smaller than or equal to the real probabilistic gap
- Given the probabilistic gap assumption, the observed probabilistic gap maintains the same ordering as the probabilistic gap. 
    - useful in extracting reliable negative examples, by selecting unlabeled examples with an observed probabilistic gap that is smaller than the smallest observed probabilistic gap of the labeled examples

## Data Assumptions
1. Negativity
    - Most simple and naive assumption is to assume that the unlabeled examples all belong to the negative class
2. Separability
    - it is assumed that the two classes of interest are naturally separated, meaning a classifier exists that can perfectly distinguish positive from negative examples
3. Smoothness
    - Examples that are close to each other are more likely to have the same label 
    - if $x_1$ and $x_2$ are similar than the probabilities $Pr(y=1|x_1)$ and $Pr(y=1|x_2)$ will be similar
    - shown with distance measures typically!

## Assumptions for an Identifiable Class Prior ($\alpha$)
- Helpful it can be estimated directly from the PU data. Unfortunately, this is an ill-defined problem because it is not identifiable: the absence of a label can be explained by either a small prior probability for the positive class or a low label frequency. In 
- Possible assumptions:
    1. Separable classes/non-overlapping distributions: positive and negative distributions are assumed not to overlap. The prior then becomes all the cases which fit that same distribution within the unlabeled data are of the positive type
    2. Positive subdomain/anchor set: Instead of requiring no overlap between the distributions, it suffices to require a subset of the instance space defined by partial attribute assignment to be purely positive. 
    3. Positive function/separability: this is a more general version, where the subdomain can be defined by any function instead of being limited to partial variable assignments. 
    4. Irreducibility: The negative distribution cannot be a mixture that contains the positive distribution (all previous imply this)

## PU Measures
- Most common metric for tuning using a PU data is based on the F1-score $F_1(y) = \frac{2pr}{p+r}$
- Under SCAR - the recall can be estimated from the PU data but the precision cannot. 
- Thus, an adaptation was created: $\frac{r^2}{Pr(y=1)}$

Computing standard evaluation metrics is possible via the SCAR assumption: 
1. by estimating the label frequency or class prior, it is possible to compute the expected number of positive examples in the unlabeled data
2. The rank distributions of the observed positives and the positive examples contained within the unlabeled data should be similiar. 

Combining the above information enables reasoning about the total number of positive examples below a given rank. --> then can derive typical metrics
(bassicaly estimate the total number of positive and negatives and use those instead of unlabeled and positive)

## PU Learning Methods: 

### Two step techniques:
**SUMMARY**: two step is basically the process of trying to choose the best possible negatives from the data then applying a semi-supervised algorithm to the data. 
#### Step 1: Choosing reliable negatives: 
- unlabeled examples that are very different from the positive examples are selected as reliable negatives with many methods being proposed (many originate from the text classification domain):
- Examples: 
    - **SPY**: some of the labeled examples are added to the unlabeled dataset. THEN A naive bayes classifier is trained, considering the unlabeled examples as negative and updated once using expectation maximization (need some negative data).
    - **1-DNF**  Strong positive features are learned by searching for features that occur more often in the positive data than in unlabeled. The reliable negative examples that do not have any strong positive features. (sometimes this only yields a few data points which is sub-optimal)
    - **Rocchio**: method builds a prototype for both the labeled and unlabeled examples. Essentially a way of creating a distance (similiar to k-means clustering or knn) which can be applied to find those pieces which sit far away. One could also use cosine similarity.
    - **PNLH**: another form of clustering
    - **PE**: graph-based method
    **Augmented Negatives**: generate new examples that are most likely negative. All the unlabeled and added examples are then initialized as negative. 

#### Step 2: (Semi-) Supervised Learning
- Any supervised method work
- Semi-supervised methods like expectation maximization on top of Naive Bayes can also incorporate the remaining unlabeled examples. 
- Custom methods have also been proposed: 
    - Iterative, SVM and LS-SVM. 
    - DILCA-KNN

#### Step 3: Classifier selection
- Expectation maximation generates a new model during every iteration. The local maximum to which EM converges might not be the best model in the sequence. Methods ahve been proposed to address this. 

### Biased Learning
- Considers PU data as a fully labeled dataset with class label noise for the negative class
- As long as the noise for the negative examples is constant, this setting makes the SCAR assumption. 

#### Version 1: Classification
- Large fraction of these methods are based on SVMs
- Biased SVM: standard SVM that penalizes misclassified positive and negative examples differently, with extra penalty when confident unlabeled examples are misclassified (what does this mean)
- noisiness of the negative data makes the learning harder - as too much importance might be given to a negative example that is actually positive. 
    - Addressed via bagging techniques or using least square SVMs 
    - Bagging SVM learns multiple biased SVM classifiers which are trained on the positive examples and a subset of the negative examples. 
        -see robust ensemble SVM which also resamples the positive examples and a subset of the negative examples. 
    - RankSVM - SVM method that minimizes a regularized margin-based pairwise loss. In this method the two classes do not get a different penalty, but the regularization parameter and threshold for classification are set by tuning on a part of SCAR
    - weighted logistic regression favors correct positive classification over correct negative classification by giving larger weights to positive examples. Positive examples are weighted by the negative class prior
- Methods also exist for clustering and matrix completion

### Incorporation of the Class Prior
#### Postprocessing
- The probability of an example being labeled is directly proportional to the probability of that example being positive with label frequency *c* as the proportionality constant. 
- I think this is essentially an adjustment made on the prevalence but I could be wrong
- ?? not exactly sure what is going on here

#### Preprocessing
- create a new dataset from a PU dataset which can be used by methods tht expect fully supervised data to train the best possible model for the PU data
- Methods
    1. Rebalancing methods - weight the data so that the classifier trained on the weighted data will give the same classification with the same target probability threshold as the traditional classifier. There is a second version of this called rank pruning (more robust to noise) which first lceans the data based on the class prior and the expected positive label noise. 
        - **CAVIAT** only appropriate when one is interested in classification on the given target threshold $\tao$
    2. Incorporation of the label probabilities
        - Proposed to duplicate the unlabeled examples to let them count partially as positive and partially as negative, with weights being assigned based on the probabilities of the unlabeled examples being positive or negative. 
    3. Empirical-Risk-Minimization based methods - find the classifier that minimizes the risk that the classifier trained on all data is equal to the one trained on a fully labeled dataset. 

#### Method Modification
- Many ML methods rely on the counts of positive and negative examples within the data. The counts can be estimated using the same rationale as was used for weighting?? which was what? 

### Relational Approaches
- a common task for relational data is to complete automatically constructed knowledge bases or networks by finding new relationships. This can be viewed as a PU learning problem
- When the SCAR assumption holds in the relational PU data then , relational versions of the classic class prior incorperation methods can be used to enable learning. 

### How to choose a method: 
Best way to choose which method to use? which assumptions are mostly likely to hold for the application at hand. 

- if separability holds --> this favors the two-step techniques
- If SCAR holds, then one would use biased learning or methods that incorporate the class prior
- If both separability and SCAR hold, the choice depends on how clearly separated the two classes are
- If the classes are separable but very close, seperating the two classes correctly is hard for two step techniques so using SCAR is more likely to be effective. 
- If the classes are very clearly seperated the two-step techniques are favored (more robust against deviations). 
- Not many methods exist for SAR and PGPU assumptions. 

## Class Prior Estimation from PU Data
Knowledge of the class prior significantly simplifies PU learning under SCAR. Thus ti can be useful to estimate it from the data directly...

### Non-traditional Classifier
- train a classifier and hold a seperate validation set, then estimate the label frequency as the average predicted probability of a label validation set example. (requires good model calibration)
- another method based on a non-traditional classifier uses the insight that the probability which is estimated by g(x) is equal to the label frequency when the true conditional class probability =1? WTF

### Partial Matching:
- assumes non-overlapping classes. It uses a density estimation method to estimate the positive distrubtion based on the labeled examples and the complete distribution based on all the data. 
- The class prior is found by minimizing the difference between the scaled positive distribution, where the scale factor is the class prior. Basically I think you are tryi
ng to scale down the real positive distribution to make the data look correct (this is your prior)

### Decision Tree Induction
- estimates the label frequency c under the positive subdomain assumption (what is this??). It makes the observation that the label frequency remains the same when considering a subdomain of the data and the that fraction of labeled examples in that subdomain provides a natural lower bound to the label freuqency. 

### ROC Approaches
- one aims to maximize the TPR while minimizing the FPR. The TPR can be calculated in PU data, by using the labeled positive set. For a given TPR, minimizing the FPR within a hypothesis space is equivalent to minimizing the probability of predicting the positive class. If a classifier f exists that minimizes the FPR to zero, then the class prior can be calculated. 
- Search subspace can be larger, so one needs strict positive subdomain assumptions --> k-kernel density estimation to approximate positive and total distributions. 

### Which one? 
- Whether a particular approach is suitable for a problem will depend on the assumptions underpinning the approach and how well they match the problem at hand.
- non-traditional classifier and some partial matching approaches make the assumption that the positive and negative examples are seperable. BUT it is unlikely that this will hold in practice. It is possible to relax this restriction via partial matching approach such that only a positive subdomain is assumed. 
- Decision Tree, ROC also make this assumption but do not provide guarantees in terms of convergence to the true estimate. 
- Empirically, the comparisons among these approaches tend to focus on idealized conditions on artificially constructed PU data. In practice, KM2 and TIcE produced the most accurate estimates on SCAR PU data. 

### Some other things...
- Most common labeling mechanism assumption is selected completely at random (SCAR) with more frameworks focusing in on the SCAR assumption (labeling mechanism depends on the attributes)
- estimation of the class prior (aka the frequency in certain conditions) and is found to be incredibly helpful. 
- HOW DOES ONE EVALUATE MODELS IF THE SCAR ASSUMPTION IS VIOLATED? aka SAR - then most of the shit fails


https://pulearn.github.io/pulearn/doc/pulearn/
https://github.com/pulearn/pulearn
https://arxiv.org/pdf/1811.04820.pdf
%Tutorials used for reference: 
%http://matlabdatamining.blogspot.com/2010/02/principal-components-analysis.html
%http://www.nlpca.org/pca-principal-component-analysis-matlab.html

%In this example, we will demonstrate the usefulness of applying principle
%component analysis (PCA) to a dataset. We will be using the Fisher Iris
%dataset. It contains measurements of 3 different species of iris flowers 
%(150 flowers total, 50 in each category, 4 predictors/measurements per example).

%The measurements include: petal length, petal width, sepal length, and
%sepal width. We are interested being able to classify an iris flower into
%one of the 3 species based on these measurements. We will first view the
%raw data. Then we will apply PCA and observe the difference it makes.

%Load the iris dataset. Two variables are now in the workspace: meas, and
%species. meas holds the attributes of each observation, and species holds
%the classification of each observation/example.
clear
load fisheriris.mat

%Now lets take a quick look at a couple plots of the raw data and observe
%patterns. We'll plot setosas as red, versicolors as blue, and virginicas
%as green. We'll plot the 4 different combinations of attributes against
%each other.
petLen = meas(:, 1);    %extract attributes individually
petWid = meas(:, 2);
sepLen = meas(:, 3);
sepWid = meas(:, 4);

figure(1)
hold on
xlabel('Petal Length')
ylabel('Petal Width')
scatter(petLen(find(contains(species,'setosa'))), petWid(find(contains(species,'setosa'))), 'r')
scatter(petLen(find(contains(species,'versicolor'))), petWid(find(contains(species,'versicolor'))), 'b')
scatter(petLen(find(contains(species,'virginica'))), petWid(find(contains(species,'virginica'))), 'g')

figure(2)
hold on
xlabel('Sepal Length')
ylabel('Sepal Width')
scatter(sepLen(find(contains(species,'setosa'))), sepWid(find(contains(species,'setosa'))), 'r')
scatter(sepLen(find(contains(species,'versicolor'))), sepWid(find(contains(species,'versicolor'))), 'b')
scatter(sepLen(find(contains(species,'virginica'))), sepWid(find(contains(species,'virginica'))), 'g')

figure(3)
hold on
xlabel('Petal Length')
ylabel('Sepal Width')
scatter(petLen(find(contains(species,'setosa'))), sepWid(find(contains(species,'setosa'))), 'r')
scatter(petLen(find(contains(species,'versicolor'))), sepWid(find(contains(species,'versicolor'))), 'b')
scatter(petLen(find(contains(species,'virginica'))), sepWid(find(contains(species,'virginica'))), 'g')

%On the last plot we are adding a small ammount of randomness to the data.
%This creates a more realistic looking scatter, and it uncovers points that
%lay on top of one another.
figure(4)
hold on
xlabel('Sepal Length')
ylabel('Petal Width')
scatter(sepLen(1:50)+randn(50,1)*0.02, petWid(1:50)+randn(50,1)*0.02, 'r')
scatter(sepLen(51:100)+randn(50,1)*0.02, petWid(51:100)+randn(50,1)*0.02, 'b')
scatter(sepLen(101:150)+randn(50,1)*0.02, petWid(101:150)+randn(50,1)*0.02, 'g')

%We've now plotted our datapoints from all 4 combinations of 2 variables out of 4. It's clear that
%there is quite a bit of overlap between classes, particularly between
%virginica and vesicolor. If you were given a new observation and it fell
%within the overlapping zones, you wouldn't know how to classify it. Once
%we apply PCA, this problem will be reduced greatly. Because we will be
%using two heavily information-loaded variables rather than any two of the
%original ones. 

%Now lets apply PCA to the dataset.
%First, lets collect some general information that will be useful later.
[n m] = size(meas)
dataMean = mean(meas)
dataStd = std(meas)

%It is important to preprocess the data a bit in PCA. Usually we subtract
%the mean from each observation, and optionally, divide each by the 
%standard deviation. Subtracting the mean centers the data on 0, and
%dividing by the standard deviation normalizes the data.
cleanData = (meas - repmat(dataMean,[n 1]))
%cleanData = cleanData ./ repmat(dataStd, [n 1]) %optional

%In order to do element-wise subtraction of the mean, we create a matrix of 
%the same size as meas where each row is the dataMeans matrix.
%We use repmat here. This is a handy function that allows you to treat your
%input matrix as a single variable to create a matrix of your matrix. Think
%of is as equivilant to ones(), but instead of a matrix of ones, its a
%matrix of your input matrix.

%The main idea behind PCA is to take the dataset from the original
%coordinate system, to a coordinate system where the variance of the data 
%is maximized. In order to this, we take the eigenfunction of the
%data's covariance matrix. The eigenvectors will be the principal component
%coefficients, and the corresponding eigenvalues correspond to the ammount
%of variance their respective principal component encapsulates.
[coeffs vars] = eig(cov(cleanData))
vars = diag(vars);   %isolate the eigenvalues (variances)

%eig does not necessarily give the eigenvalue-eigenvector pairs in a particular order. We need to sort them in decreasing order so that
%the principal components with the largest ammount of variance come first
[vars, sortedidx]=sort(vars,'descend');
coeffs = coeffs(:, sortedidx)
vars

%Now we are set up to take the data from it's original coordinate system to
%the PCA coordinate system. All we have to do is multiply the data matrix
%with the principal component coefficient matrix.
pcaData = cleanData * coeffs

%So now we've obtained the principal components of the data. The columns of
%pcaData correspond to the principal components (col1 = principal
%component1 ect). Lets look at some plots to illustrate what we've done.

%First, we look at the first 3 principal components (PC) individually. What we
%see is that the overlapping area of the classes gets wider as we progress.
%This corresponds to the ammount of variance captured by each PC.

figure(5)
hold on
title('Principal Component 1')
scatter(pcaData(1:50, 1), pcaData(1:50, 1), 'r')
scatter(pcaData(51:100, 1), pcaData(51:100, 1), 'b')
scatter(pcaData(101:150, 1), pcaData(101:150, 1), 'g')

figure(6)
hold on
title('Principal Component 2')
scatter(pcaData(1:50, 2), pcaData(1:50, 2), 'r')
scatter(pcaData(51:100, 2), pcaData(51:100, 2), 'b')
scatter(pcaData(101:150, 2), pcaData(101:150, 2), 'g')

figure(7)
hold on
title('Principal Component 3')
scatter(pcaData(1:50, 3), pcaData(1:50, 3), 'r')
scatter(pcaData(51:100, 3), pcaData(51:100, 3), 'b')
scatter(pcaData(101:150, 3), pcaData(101:150, 3), 'g')

%Lastly, we look at the first 2 PCs plotted against each other. In this
%coordinate space, we can see more clearly defined and seperated classes.

figure(8)
hold on
xlabel('PC1')
ylabel('PC2')
scatter(pcaData(1:50, 1), pcaData(1:50, 2), 'r')
scatter(pcaData(51:100, 1), pcaData(51:100, 2), 'b')
scatter(pcaData(101:150, 1), pcaData(101:150, 2), 'g')

%Additionally, we can calculate the variance that each PC captures as a
%percentage. This is a useful and often impressive calculation. We do this
%by dividing the variance of each PC by the total variance of all PCs.
%var(pcaData) / sum(var(pcaData))
vars / sum(vars)

%Each column in this matrix corresponds to the variance captured by PC.
%Amazingly, PC1 captures 92% of the variance by itself. If we keep k PCs,
%it is important to add these percentages up to know the proportion of
%variance explained with k variables. This plot will gradually plateau indicating that we have 
%sufficiently many principal components. 

figure(9)
bar( cumsum(vars) / sum(vars))
ylabel('Proportion of variance explained')
xlabel('#selected PCs')

%So, we applied PCA to the data and got a graph that seems to make more
%sense than the raw data. Now we want to put it to use. 
%Say someone wants us to use our PCA technique to classify a few iris
%flowers for them and they give us petal/sepal dimensions as follows.
testFlowers = [6.1 2.7 5.0 1.6; 5.3 3.2 2.1 0.4; 7.75 3.5 6.1 2.0]

%The process to classify these new examples is simple. We apply PCA to each
%example, and determine the class to be the same of it's nearest neighbor.
%So lets apply PCA.

%Don't forget to subtract the mean of the original data.
[tn tm] = size(testFlowers);
cleanTestFlowers = testFlowers - repmat(dataMean,[tn 1])

%After cleaning, we take the test data to the PCA coordinate space by 
%multiplying it by our original PCA coefficients.
pcaTestData = cleanTestFlowers * coeffs

%Now we can plot the test data over our original PCA graph to see where it
%lands. Our first test point is a magenta astrisk, the second is the plus,
%and the last is the x.
figure(8)
scatter(pcaTestData(1, 1), pcaTestData(1, 2), '*', 'm')
scatter(pcaTestData(2, 1), pcaTestData(2, 2), '+', 'm')
scatter(pcaTestData(3, 1), pcaTestData(3, 2), 'x', 'm')

%Now get the closest neighbor and classify. We do that by getting the
%distances from our point of interest to every other point, and taking the
%minimum. The point responsible for that minimum is our nearest neighbor,
%and tells us which class our new point belongs to.
dists = zeros(n, tn);
for numTPoints = 1:tn
    for i = 1:n
        dists(i, numTPoints) = norm(pcaTestData(numTPoints, :) - pcaData(i, :));
    end
end

testClasses = zeros(tn, 1);
for i = 1:tn
    [~, index] = min(dists(:,i));
    if index <= 50
        testClasses(i) = 1
    elseif index > 50 && index <= 100
        testClasses(i) = 2
    else
        testClasses(i) = 3
    end
end

%Our test points have now been classified as 2 (versicolor or red on the
%graphs), 1 (setosa or blue), and 3 (virginica or green). By going back to
%look at our PC1 vs PC2 graph, we can see that we successfully classified
%our test set to the nearest neighbor.


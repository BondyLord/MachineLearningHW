package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.PriorityQueue;

class DistanceCalculator {
    /**
     * We leave it up to you wheter you want the distance method to get all relevant
     * parameters(lp, efficient, etc..) or have it has a class variables.
     *
     * @param distanceMethod
     */
    public double distance(Instance one, Instance two, Knn.lpDistance distanceMethod) {
        double distance = 0;
        switch (distanceMethod) {
            case Infinity:
                distance = lInfinityDistance(one, two);
                break;
            case One:
                distance = lpDisatnce(one, two, 1);
                break;
            case Two:
                distance = lpDisatnce(one, two, 2);
                break;
            case Three:
                distance = lpDisatnce(one, two, 3);
                break;
            case EfficientOne:
                distance = efficientLpDisatnce(one, two, 1);
                break;
            case EfficientTwo:
                distance = efficientLpDisatnce(one, two, 2);
                break;
            case EfficientThree:
                distance = efficientLpDisatnce(one, two, 3);
                break;
            case EfficientInfinity:
                distance = efficientLInfinityDistance(one, two);
                break;
        }
        return distance;
    }

    /**
     * Returns the Lp distance between 2 instances.
     *
     * @param one
     * @param two
     */
    private double lpDisatnce(Instance one, Instance two, double powerOf) {
        double sumOfDistances = 0;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            double attributeVal = one.value(i);
            double attributeVal2 = two.value(i);
            double powAttribute = Math.pow(attributeVal - attributeVal2, powerOf);
            sumOfDistances += Math.abs(powAttribute);
        }
        double distanceBetweenInstances = Math.pow(sumOfDistances, 1 / powerOf);
        return distanceBetweenInstances;
    }

    /**
     * Returns the L infinity distance between 2 instances.
     *
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        double biggestDistance = 0;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            double attributeVal = one.value(i);
            double attributeVal2 = two.value(i);
            double currentDistance = Math.abs(attributeVal - attributeVal2);
            if (biggestDistance < currentDistance) {
                biggestDistance = currentDistance;
            }
        }
        return biggestDistance;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient
     * distance check.
     *
     * @param one
     * @param two
     * @return
     */
    private double efficientLpDisatnce(Instance one, Instance two,double powerOf) {
        // TODO implement
        double threshold = Double.MAX_VALUE;
        double sumOfDistances = 0;
        double distance = 0;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            double attributeVal = one.value(i);
            double attributeVal2 = two.value(i);
            distance = Math.pow(attributeVal - attributeVal2, powerOf);
            sumOfDistances += Math.abs(distance);
            if (sumOfDistances > Math.pow(distance, powerOf)) {
                break;
            }
        }
        double distanceBetweenInstances = Math.pow(sumOfDistances, 1 / powerOf);
        return distanceBetweenInstances;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient
     * distance check.
     *
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two) {
        // TODO implement
        return 0.0;
    }
}

public class Knn implements Classifier {

    // A class that describes an instance and his distance from the instance we want to classify
    public class MyInstance implements Comparable<MyInstance> {
        private Instance instance;
        private double distanceFromSource;

        public MyInstance(Instance instance, double distanceFromSource) {
            this.instance = instance;
            this.distanceFromSource = distanceFromSource;
        }

        public double GetDistance() {
            return this.distanceFromSource;
        }

        public int compareTo(MyInstance other) {
            if (this.distanceFromSource < other.distanceFromSource) {
                return -1;
            } else if (this.distanceFromSource > other.distanceFromSource) {
                return 1;
            }
            return 0;
        }
    }

    // K's variable and functions
    private int k;

    public int getK() {
        return this.k;
    }

    // DistanceCheck's variable and functions
    public enum DistanceCheck {
        Regular, Efficient
    }

    private DistanceCheck distanceCheck;

    public DistanceCheck getDistanceCheck() {
        return this.distanceCheck;
    }

    // WeightingScheme's variable and functions
    public enum e_WeightingScheme {
        Uniform, Weighted
    }

    ;
    private e_WeightingScheme weightingScheme;

    public e_WeightingScheme getWeightingScheme() {
        return this.weightingScheme;
    }

    // lpDistance's variable and functions
    public enum lpDistance {
        One, Two, Three, Infinity, EfficientOne,EfficientTwo,EfficientThree, EfficientInfinity
    }

    ;
    private lpDistance distanceMethod;

    public lpDistance getLpDistanceMethod() {
        return this.distanceMethod;
    }

    public void setDistanceMethod(lpDistance distanceMethod) {
        this.distanceMethod = distanceMethod;
    }

    // Other variables and functions
    private Instances m_trainingInstances;

    /**
     * A function that's responsible for finding the best hyper parameters and
     * returns the Corresponding error for those parameters
     *
     * @param arg0
     * @return the error for the best hyper parameters
     */
    public double findBestHyperParametersAndError(Instances arg0) {

        this.m_trainingInstances = arg0;
        double MinError = Double.MAX_VALUE;
        int best_k = 0;
        e_WeightingScheme best_scheme = null;
        lpDistance best_distance = null;
        // runs over the possible k's
        for (int i = 1; i <= 20; i++) {

            this.k = i;

            // runs over the possible lpDistances
            for (lpDistance disMethod : lpDistance.values()) {
                if (disMethod.name().startsWith("Efficient")) {
                    break;
                }
                this.distanceMethod = disMethod;

                // runs over the possible weightSchemes
                for (e_WeightingScheme weightScheme : e_WeightingScheme.values()) {

                    this.weightingScheme = weightScheme;
                    double current_error = crossValidationError(arg0, 10);

                    if (current_error < MinError) {
                        MinError = current_error;
                        best_k = this.k;
                        best_distance = this.distanceMethod;
                        best_scheme = this.weightingScheme;
                    }
                }
            }

        }

        this.k = best_k;
        this.distanceMethod = best_distance;
        this.weightingScheme = best_scheme;

        return MinError;
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     *
     * @param insances     Instances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances instances, int num_of_folds) {
        Instances[] kFold = makeKFold(this.m_trainingInstances, num_of_folds);
        double SumErrorsOnKFold = 0;

        for (int i = 0; i < kFold.length; i++) {
            SumErrorsOnKFold += averageErrorTheIFold(kFold, i, instances);
        }

        double averageErrorOnKFold = SumErrorsOnKFold / kFold.length;
        return averageErrorOnKFold;
    }

    /**
     * divides instances to k folds
     *
     * @param Instances instances - A group of instance to split
     * @param int       numberOfFolds - the number of folds to split the data to
     * @return Instances[] kFold - an array of size numberOfFolds of with the instances divided in them equally
     */
    public Instances[] makeKFold(Instances instances, int numberOfFolds) {
        Instances[] kFold = new Instances[numberOfFolds];
        int cell = 0;
        Instances instancesToUse = new Instances(instances);
        // creating new kFold
        while (instancesToUse.numInstances() > 0) {
            int currentInstance = (int) (Math.random() * instancesToUse.numInstances());
            if (kFold[cell] == null) {
                // creating new instances cell
                kFold[cell] = new Instances(instancesToUse, 0);
            }

            kFold[cell].add(instancesToUse.remove(currentInstance));
            cell++;

            // verify that we are not deviating from the array
            cell %= numberOfFolds;
        }
        return kFold;
    }

    /**
     * calculate the average error on the i fold
     *
     * @param kFold
     * @param testCell
     * @param instances
     * @return error for the current division
     */
    private double averageErrorTheIFold(Instances[] kFold, int testCell, Instances instances) {
        Instances toLearnFrom = KMinusOneDataToLearnFrom(kFold, testCell, instances);
        Instances originalData = new Instances(this.m_trainingInstances);
        this.m_trainingInstances = toLearnFrom;
        double error = calcAvgError(kFold[testCell]);
        // Changing back my m_trainingInstances to be the original dataset
        this.m_trainingInstances = originalData;
        return error;
    }

    /**
     * returns the instances to learn from (all instances but the test fold)
     *
     * @param kFold
     * @param testCell
     * @param instances
     * @return
     */
    private Instances KMinusOneDataToLearnFrom(Instances[] kFold, int testCell, Instances instances) {
        Instances toLearnFrom = new Instances(instances, 0);

        for (int i = 0; i < kFold.length; i++) {
            // skips on the testing fold
            if (i == testCell) {
                continue;
            }

            toLearnFrom.addAll(kFold[i]);
        }

        return toLearnFrom;
    }

    /**
     * Calculates the average error on a give set of instances. The average error is
     * the average absolute error between the target value and the predicted value
     * across all instances.
     *
     * @param insatnces
     * @return
     */
    public double calcAvgError(Instances testingInstances) {
        double sumError = 0;
        for (int i = 0; i < testingInstances.numInstances(); i++) {
            double error = Math.abs(testingInstances.instance(i).classValue() - classifyInstance(testingInstances.instance(i)));
            sumError += error;
        }
        sumError /= testingInstances.numInstances();
        return sumError;
    }

    @Override
    public double classifyInstance(Instance instance) {
        Instances kNearestNeighbors = findNearestNeighbors(instance);
        double classifier = 0;

        if (this.weightingScheme.equals(e_WeightingScheme.Weighted)) {
            classifier = getWeightedAverageValue(kNearestNeighbors, instance);
        } else {
            classifier = getAverageValue(kNearestNeighbors);
        }
        return classifier;
    }

    /**
     * Finds the k nearest neighbors.
     *
     * @param instance
     */
    public Instances findNearestNeighbors(Instance instance) {
        PriorityQueue<MyInstance> minHeapKneighbors = new PriorityQueue<MyInstance>();

        // Add all training instances to minHeapKneighbors as MyInstance objects
        for (int i = 0; i < this.m_trainingInstances.numInstances(); i++) {
            DistanceCalculator dc = new DistanceCalculator();
            double distanceForCurrentInstance = dc.distance(instance, this.m_trainingInstances.instance(i), this.distanceMethod);
            minHeapKneighbors.add(new MyInstance(this.m_trainingInstances.instance(i), distanceForCurrentInstance));
        }

        // Add the k nearest neighbors to KNearestNeighbors
        Instances KNearestNeighbors = new Instances(this.m_trainingInstances, 0);
        for (int i = 0; i < this.k; i++) {
            KNearestNeighbors.add(minHeapKneighbors.remove().instance);
        }

        return KNearestNeighbors;
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     *
     * @param kNearestNeighbors
     * @param
     * @return
     */
    public double getAverageValue(Instances kNearestNeighbors) {
        return kNearestNeighbors.meanOrMode(kNearestNeighbors.classAttribute());
    }

    /**
     * Calculates the weighted average of the target values of all the elements in
     * the collection with respect to their distance from a specific instance.
     *
     * @param kNearestNeighbors
     * @param instance
     * @return
     */
    public double getWeightedAverageValue(Instances kNearestNeighbors, Instance instance) {
        //TODO: validate output is correct
        double average = 0.0;
        DistanceCalculator distanceCalculator = new DistanceCalculator();
        double distance = 0.0, sumDistances = 0.0, sumWightedDistances = 0.0, wi = 0.0;
        for (Instance neighbor : kNearestNeighbors) {
            distance = distanceCalculator.distance(neighbor, instance, distanceMethod);
            if (distance == 0) {
                return neighbor.classValue();
            }
            wi = 1.0 / Math.pow(distance, 2);
            if (wi > 0) {
                sumDistances += wi;
                sumWightedDistances += wi * neighbor.classValue();
            }
        }
        average = sumWightedDistances / sumDistances;
        return average;
    }

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for
     * later use in the prediction.
     *
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
        //TODO implement ?
    }

    /**
     * Returns the knn prediction on the given instance.
     *
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        return 0.0;
    }

    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }
}

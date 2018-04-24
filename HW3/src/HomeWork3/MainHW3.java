package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import HomeWork3.Knn.e_WeightingScheme;
import HomeWork3.Knn.lpDistance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class MainHW3 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		Instances originalDataset = loadData("auto_price.txt");
		Instances scaledDataset = FeatureScaler.scaleData(originalDataset);
		Knn originalDatasetKnn = new Knn();
		double original_error = originalDatasetKnn.findBestHyperParametersAndError(originalDataset);
		int original_k =  originalDatasetKnn.getK();
		lpDistance original_lp_distance_method = originalDatasetKnn.getLpDistanceMethod();
		e_WeightingScheme original_weighting_scheme = originalDatasetKnn.getWeightingScheme();
		
		System.out.println("----------------------------");
		System.out.println("Results for original dataset:");
		System.out.println("----------------------------");
		System.out.println("Cross validation error with K = " + original_k + ", lp = " + original_lp_distance_method + ", majority function");
		System.out.println("= " + original_weighting_scheme + " for auto_price data is: " + original_error);
		
		
		Knn scaledDatasetKnn = new Knn();
		double scaled_error = scaledDatasetKnn.findBestHyperParametersAndError(scaledDataset);
		int scaledl_k =  scaledDatasetKnn.getK();
		lpDistance scaled_lp_distance_method = scaledDatasetKnn.getLpDistanceMethod();
		e_WeightingScheme scaled_weighting_scheme = scaledDatasetKnn.getWeightingScheme();
		
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Results for scaled dataset:");
		System.out.println("----------------------------");
		System.out.println("Cross validation error with K = " + scaledl_k + ", lp = " + scaled_lp_distance_method + ", majority function");
		System.out.println("= " + scaled_weighting_scheme + " for auto_price data is: " + scaled_error);

		int [] numberOfFolds = {originalDataset.numInstances(), 50, 10, 5, 3};
		
		for (int i = 0; i < numberOfFolds.length; i++) {
			System.out.println();
			System.out.println();
			System.out.println("----------------------------");
			System.out.println("Results for " +numberOfFolds[i] + " folds:");
			System.out.println("----------------------------");
			Knn autoPriceDataset = new Knn();
			double crossValidationError = 0;
//			crossValidationError = autoPriceDataset.findBestHyperParametersAndError(autoPriceDataset);
			System.out.println("Cross validation error of regular knn on auto_price dataset is <error> and");
			System.out.println("the average elapsed time is <average_elapsed_time_in_nano_seconds>");
			System.out.println("The total elapsed time is: <total_elapsed_time_in_nano_seconds>");
			System.out.println();
			System.out.println("oss validation error of efficient knn on auto_price dataset is <error> and");
			System.out.println("the average elapsed time is <average_elapsed_time_in_nano_seconds>");
			System.out.println("The total elapsed time is: <total_elapsed_time_in_nano_seconds>");	
		}	
		
	}
	
	public static class FeatureScaler {
		/**
		 * Returns a scaled version (using standarized normalization) of the given dataset.
		 * @param instances The original dataset.
		 * @return A scaled instances object.
		 * @throws Exception 
		 */
		public static Instances scaleData(Instances instances) throws Exception {
			Standardize filter = new Standardize();
			filter.setInputFormat(instances);
			instances = Filter.useFilter(instances, filter);
			return instances;
		}
	}
	
}


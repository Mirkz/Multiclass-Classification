package feedForwardNeuralNetwork;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;

public class FeedForwardNeuralNetwork {
	
	// configuration parameters
	final static int NoOfNodes = 24;
	final static int NoOfIterations = 1000;
	final static double alphaZero = 1.49;
	final static boolean tanh = true;
	final static boolean atan = false;	
	final static double perc = 0.8;		// regularisation (insert percentage of hidden nodes not to be dropped)

	
	
	final static String TRAIN_FILE_PATH = "fashion-mnist_train.csv";
	final static String TEST_FILE_PATH = "fashion-mnist_test.csv";
	final static int PIXELS = 784;
	final static int TRAIN_DIM = 60000;
	final static int TEST_DIM = 10000;
	final static int OUTPUT_DIM = 10;

	static double[][] trainData = new double[PIXELS][TRAIN_DIM];
	static int[] trainLabels = new int[TRAIN_DIM];
	static double[][] testData = new double[PIXELS][TEST_DIM];
	static int[] testLabels = new int[TEST_DIM];

	static SimpleMatrix w1 = new SimpleMatrix(PIXELS, NoOfNodes);		// weights of hidden layer
	static SimpleMatrix w2 = new SimpleMatrix(NoOfNodes, OUTPUT_DIM);	// weights of output layer
	static SimpleMatrix b1 = new SimpleMatrix(NoOfNodes, TRAIN_DIM);	// biases of hidden layer
	static SimpleMatrix b2 = new SimpleMatrix(OUTPUT_DIM, TRAIN_DIM);	// biases of output layer
	
	

	public static void main(String[] args) {
		System.out.print("Loading dataset... ");
		loadCsvFiles(TRAIN_FILE_PATH, trainData, trainLabels);
		loadCsvFiles(TEST_FILE_PATH, testData, testLabels);
		System.out.println("done!");
		
		train();
		
		test();
	}
	
	

	private static void test() {
		SimpleMatrix a0 = new SimpleMatrix(testData);

		// adapt dimensions of bias matrices to TEST_DIM
		SimpleMatrix b1_test = new SimpleMatrix(NoOfNodes, TEST_DIM);
		SimpleMatrix b2_test = new SimpleMatrix(OUTPUT_DIM, TEST_DIM);
		for (int row = 0; row < NoOfNodes; row++) {
			for (int col = 0; col < TEST_DIM; col++) {
				b1_test.set(row, col, b1.get(row, col));
			}
		}
		for (int row = 0; row < OUTPUT_DIM; row++) {
			for (int col = 0; col < TEST_DIM; col++) {
				b2_test.set(row, col, b2.get(row, col));
			}
		}
		
		

		// forward propagation
		SimpleMatrix z1 = new SimpleMatrix(NoOfNodes, TEST_DIM);
		z1 = w1.transpose().mult(a0).plus(b1_test);
		SimpleMatrix a1 = new SimpleMatrix(NoOfNodes, TEST_DIM);
		for (int r = 0; r < NoOfNodes; r++) {
			for (int c = 0; c < TEST_DIM; c++) {
				double argument = z1.get(r, c);
				a1.set(r, c, Math.tanh(argument));
			}
		}

		SimpleMatrix z2 = new SimpleMatrix(OUTPUT_DIM, TEST_DIM);
		z2 = w2.transpose().mult(a1).plus(b2_test);
		SimpleMatrix a2 = new SimpleMatrix(OUTPUT_DIM, TEST_DIM);
		int[] prevision = new int[TEST_DIM];
		for (int c = 0; c < TEST_DIM; c++) {
			double highestProbability = -1;
			int highestClass = -1;

			double denominator = 0;
			for (int r = 0; r < OUTPUT_DIM; r++) {
				denominator += Math.pow(Math.E, z2.get(r, c));
			}
			for (int r = 0; r < OUTPUT_DIM; r++) {
				double numerator = Math.pow(Math.E, z2.get(r, c));
				a2.set(r, c, numerator / denominator);

				if (a2.get(r, c) > highestProbability) {
					highestProbability = a2.get(r, c);
					highestClass = r;
				}
			}

			prevision[c] = highestClass;
		}
		
		

		// accuracy
		int success = 0;
		for (int sample = 0; sample < TEST_DIM; sample++) {
			if (prevision[sample] == testLabels[sample]) {
				success++;
			}
			System.out.println("prediction: " + prevision[sample] + " real class: " + testLabels[sample]);
		}

		System.out.println("\n" + success + " correct predictions");
		System.out.println("final accuracy: " + (double) success / TEST_DIM);
	}
	
	

	private static void train() {
		
		// fill w1 with random double values in (-1;1) 
		Random rand = new Random();
		for (int col = 0; col < NoOfNodes; col++) {
			for (int row = 0; row < PIXELS; row++) {
				double randomValue = rand.nextDouble();
				if (rand.nextInt(2) > 0) {
					randomValue *= -1;
				}
				w1.set(row, col, randomValue);
			}
		}

		// fill w2 with random double values in (-1;1)
		for (int col = 0; col < OUTPUT_DIM; col++) {
			for (int row = 0; row < NoOfNodes; row++) {
				double randomValue = rand.nextDouble();
				if (rand.nextInt(2) > 0) {
					randomValue *= -1;
				}
				w2.set(row, col, randomValue);
			}
		}

		// Y is the matrix with the real probabilities for each class, and for each sample (either 0 or 1).
		SimpleMatrix y = new SimpleMatrix(OUTPUT_DIM, TRAIN_DIM);
		for (int col = 0; col < TRAIN_DIM; col++) {
			for (int row = 0; row < OUTPUT_DIM; row++) {
				if (trainLabels[col] == row) {
					y.set(row, col, 1);
				} else {
					y.set(row, col, 0);
				}
			}
		}

		
		
		double[] costValues = new double[NoOfIterations];
		int newNoOfNodes = (int) (NoOfNodes * perc);
		int NoOfNodesToDrop = NoOfNodes - newNoOfNodes;
		System.out.println("hidden nodes dropped at each iteration: " + NoOfNodesToDrop);
		System.out.println("hidden nodes left at each iteration: " + newNoOfNodes + "\n");
		
		

		for (int iter = 1; iter <= NoOfIterations; iter++) {
			double alpha = alphaZero * (1001 - iter) / NoOfIterations;			
			
			// regularisation - drop a few hidden nodes, selected randomly at each iteration
			// dropped nodes are stored in nodesToDrop to avoid dropping the same node more than once
			// if a node has not been dropped, then isNodeAvailable[node] is true. Otherwise, isNodeAvailable[node] is false
			boolean[] isNodeAvailable = new boolean[NoOfNodes];
			for (int n = 0; n < NoOfNodes; n++) {
				isNodeAvailable[n] = true;
			}
			int[] nodesToDrop = new int[NoOfNodesToDrop];
			for (int n = 0; n < NoOfNodesToDrop; n++) {
				nodesToDrop[n] = -1;
			}
			for (int n = 0; n < NoOfNodesToDrop; n++) {
				int node = -1;
				boolean wasDropped;
				
				// iterate until a node that has not been dropped yet is found
				do {
					wasDropped = false;
					node = rand.nextInt(NoOfNodes);
					for (int m = 0; m < n; m++) {
						if (node == nodesToDrop[m]) {
							wasDropped = true;
							break;
						}
					}
				} while (wasDropped);
				
				nodesToDrop[n] = node;
				isNodeAvailable[node] = false;
			}
			
			SimpleMatrix newW1 = new SimpleMatrix(PIXELS, newNoOfNodes);
			int newCol = 0;
			for (int col = 0; col < NoOfNodes; col++) {
				if (isNodeAvailable[col]) {
					for (int row = 0; row < PIXELS; row++) {
						newW1.set(row, newCol, w1.get(row, col));
					}
					newCol++;
				}
			}

			SimpleMatrix newW2 = new SimpleMatrix(newNoOfNodes, OUTPUT_DIM);
			int newRow = 0;
			for (int row = 0; row < NoOfNodes; row++) {
				if (isNodeAvailable[row]) {
					for (int col = 0; col < OUTPUT_DIM; col++) {
						newW2.set(newRow, col, w2.get(row, col));
					}
					newRow++;
				}
			}

			SimpleMatrix newB1 = new SimpleMatrix(newNoOfNodes, TRAIN_DIM);
			newRow = 0;
			for (int row = 0; row < NoOfNodes; row++) {
				if (isNodeAvailable[row]) {
					for (int col = 0; col < TRAIN_DIM; col++) {
						newB1.set(newRow, col, b1.get(row, col));
					}
					newRow++;
				}
			}
			
			String nodesToDropString = "";
			for (int node : nodesToDrop) {
				if(!nodesToDropString.isEmpty())
					nodesToDropString += ", ";
				nodesToDropString += node;
			}
			System.out.println("Iteration " + iter + " of " + NoOfIterations + " - hidden nodes dropped: " + nodesToDropString);
			
			
			// forward propagation
			SimpleMatrix a0 = new SimpleMatrix(trainData);
			SimpleMatrix z1 = new SimpleMatrix(newNoOfNodes, TRAIN_DIM);
			z1 = newW1.transpose().mult(a0).plus(newB1);
			SimpleMatrix a1 = new SimpleMatrix(newNoOfNodes, TRAIN_DIM);
			for (int r = 0; r < newNoOfNodes; r++) {
				for (int c = 0; c < TRAIN_DIM; c++) {
					double argument = z1.get(r, c);
					if (tanh)
						a1.set(r, c, Math.tanh(argument));
					else if (atan)
						a1.set(r, c, Math.atan(argument));
				}
			}

			SimpleMatrix z2 = new SimpleMatrix(OUTPUT_DIM, TRAIN_DIM);
			z2 = newW2.transpose().mult(a1).plus(b2);
			SimpleMatrix a2 = new SimpleMatrix(OUTPUT_DIM, TRAIN_DIM);
			for (int c = 0; c < TRAIN_DIM; c++) {
				double denominator = 0;
				for (int r = 0; r < OUTPUT_DIM; r++) {
					denominator += Math.pow(Math.E, z2.get(r, c));
				}
				for (int r = 0; r < OUTPUT_DIM; r++) {
					double numerator = Math.pow(Math.E, z2.get(r, c));
					a2.set(r, c, numerator / denominator);
				}
			}
			
			

			// loss and cost
			double lossAccumulation = 0;
			for (int r = 0; r < OUTPUT_DIM; r++) {
				for (int c = 0; c < TRAIN_DIM; c++) {
					double output = y.get(r, c);
					double prevision = a2.get(r, c);
					double loss = -(output * Math.log(prevision) + (1 - output) * Math.log(1 - prevision));
					lossAccumulation += loss;
				}
			}
			double cost = lossAccumulation / TRAIN_DIM;
			System.out.println("alpha = " + alpha + ", cost = " + cost + "\n");
			costValues[iter - 1] = cost;
			
			

			// gradients
			SimpleMatrix ones = new SimpleMatrix(TRAIN_DIM, 1);
			for (int i = 0; i < TRAIN_DIM; i++) {
				ones.set(i, 0, 1);
			}
			SimpleMatrix dz2 = new SimpleMatrix(OUTPUT_DIM, TRAIN_DIM);
			dz2 = a2.minus(y);
			SimpleMatrix dw2 = new SimpleMatrix(OUTPUT_DIM, newNoOfNodes);
			dw2 = dz2.mult(a1.transpose()).divide(TRAIN_DIM);
			SimpleMatrix db2 = new SimpleMatrix(OUTPUT_DIM, 1);
			db2 = dz2.mult(ones).divide(TRAIN_DIM);

			SimpleMatrix w2dz2 = new SimpleMatrix(newNoOfNodes, TRAIN_DIM);
			w2dz2 = newW2.mult(dz2);
			SimpleMatrix g = new SimpleMatrix(newNoOfNodes, TRAIN_DIM);
			for (int r = 0; r < newNoOfNodes; r++) {
				for (int c = 0; c < TRAIN_DIM; c++) {
					double argument = z1.get(r, c);
					g.set(r, c, 1 - Math.pow(Math.tanh(argument), 2));
				}
			}
			SimpleMatrix dz1 = new SimpleMatrix(newNoOfNodes, TRAIN_DIM);
			dz1 = w2dz2.elementMult(g);
			SimpleMatrix dw1 = new SimpleMatrix(newNoOfNodes, PIXELS);
			dw1 = dz1.mult(a0.transpose()).divide(TRAIN_DIM);
			SimpleMatrix db1 = new SimpleMatrix(newNoOfNodes, 1);
			db1 = dz1.mult(ones).divide(TRAIN_DIM);
			
			

			// update
			newW1 = newW1.minus(dw1.transpose().scale(alpha));
			newW2 = newW2.minus(dw2.transpose().scale(alpha));
			SimpleMatrix temp = db1;
			db1 = new SimpleMatrix(newNoOfNodes, TRAIN_DIM);
			for (int r = 0; r < newNoOfNodes; r++) {
				for (int c = 0; c < TRAIN_DIM; c++) {
					db1.set(r, c, temp.get(r, 0));
				}
			}
			temp = db2;
			db2 = new SimpleMatrix(OUTPUT_DIM, TRAIN_DIM);
			for (int r = 0; r < OUTPUT_DIM; r++) {
				for (int c = 0; c < TRAIN_DIM; c++) {
					db2.set(r, c, temp.get(r, 0));
				}
			}
			newB1 = newB1.minus(db1.scale(alpha));

			newCol = 0;
			for (int col = 0; col < NoOfNodes; col++) {
				if (isNodeAvailable[col]) {
					for (int row = 0; row < PIXELS; row++) {
						w1.set(row, col, newW1.get(row, newCol));
					}
					newCol++;
				}
			}

			newRow = 0;
			for (int row = 0; row < NoOfNodes; row++) {
				if (isNodeAvailable[row]) {
					for (int col = 0; col < OUTPUT_DIM; col++) {
						w2.set(row, col, newW2.get(newRow, col));
					}
					newRow++;
				}
			}

			newRow = 0;
			for (int row = 0; row < NoOfNodes; row++) {
				if (isNodeAvailable[row]) {
					for (int col = 0; col < TRAIN_DIM; col++) {
						b1.set(row, col, newB1.get(newRow, col));
					}
					newRow++;
				}
			}

			b2 = b2.minus(db2.scale(alpha));
		}
		
		// plot learning curve
		plot(costValues);
	}
	
	

	private static void plot(double[] costValues) {
		List<Double> validValues = new ArrayList<>();
		List<Double> iterations = new ArrayList<>();

		double iteration = 0;
		for (double j : costValues) {
			if (Double.isFinite(j)) {
				validValues.add(Double.valueOf(j));
				iterations.add(Double.valueOf(iteration));
			}
			iteration++;
		}

		XYChart chart = QuickChart.getChart("Learning curve", "iteration", "cost", "cost(iteration)", iterations,
				validValues);

		new SwingWrapper(chart).displayChart();
	}

	
	
	private static void loadCsvFiles(String path, double[][] data, int[] labels) {
		String line = "";
		String cvsSplitBy = ",";

		try (BufferedReader br = new BufferedReader(new FileReader(path))) {
			br.readLine();	//avoid column headlines
			int row = 0;
			while ((line = br.readLine()) != null) {
				String[] sample = line.split(cvsSplitBy);
				for (int col = 0; col < PIXELS + 1; col++) {
					double val = Double.parseDouble(sample[col]);
					if (col == 0) {
						labels[row] = (int) val;
					} else {
						val /= 255;	// normalise
						data[col - 1][row] = val;
					}
				}
				row++;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}

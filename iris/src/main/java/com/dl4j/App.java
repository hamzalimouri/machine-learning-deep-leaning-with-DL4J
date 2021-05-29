package com.dl4j;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class App {
    public static void main(String[] args) throws Exception {
        int batchSize = 1;
        int outputSize = 3;
        int classIndex = 4;
        double learninRate = 0.001;
        int inputSize = 4;
        int numHiddenNodes = 10;
        int nEpochs = 100;
        String[] labels = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().seed(123)
                .updater(new Adam(learninRate)).list()
                .layer(0,
                        new DenseLayer.Builder().nIn(inputSize).nOut(numHiddenNodes).activation(Activation.SIGMOID)
                                .build())
                .layer(1,
                        new OutputLayer.Builder().nIn(numHiddenNodes).nOut(outputSize)
                                .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                                .activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        // System.out.println(configuration.toJson());
        UIServer uiServer = UIServer.getInstance();
        InMemoryStatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);

        model.setListeners(new StatsListener(inMemoryStatsStorage));

        File fileTrain = new ClassPathResource("iris-train.csv").getFile();

        RecordReader recordReaderTrain = new CSVRecordReader();
        recordReaderTrain.initialize(new FileSplit(fileTrain));
        DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, classIndex,
                outputSize);
        for (int i = 0; i < nEpochs; i++) {
            model.fit(dataSetIteratorTrain);
        }

        File fileTest = new ClassPathResource("irisTest.csv").getFile();
        RecordReader recordReaderTest = new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(fileTest));
        DataSetIterator dataSetIteratorTest = new RecordReaderDataSetIterator(recordReaderTest, batchSize, classIndex,
                outputSize);
        Evaluation evaluation = new Evaluation(outputSize);

        while (dataSetIteratorTest.hasNext()) {
            DataSet dataSet = dataSetIteratorTest.next();
            INDArray features = dataSet.getFeatures();
            INDArray testLabels = dataSet.getLabels();
            INDArray predictedLabels = model.output(features);
            evaluation.eval(testLabels, predictedLabels);
        }
        System.out.println(evaluation.stats());
        INDArray input = Nd4j.create(new double[][] { { 5.1, 3.5, 1.4, 0.2 }, { 4.9, 3.0, 1.4, 0.2 },
                { 6.7, 3.1, 4.4, 1.4 }, { 5.6, 3.0, 4.5, 1.5 }, { 6.0, 3.0, 4.8, 1.8 }, { 6.9, 3.1, 5.4, 2.1 } });
        System.out.println("**************");
        INDArray output = model.output(input);
        INDArray classes = output.argMax(1);
        System.out.println(output);
        System.out.println("-----------------");
        System.out.println(classes);
        System.out.println("****************");

        int[] predictions = classes.toIntVector();
        for (int i = 0; i < predictions.length; i++) {
            System.out.println(labels[predictions[i]]);
        }
    }
}

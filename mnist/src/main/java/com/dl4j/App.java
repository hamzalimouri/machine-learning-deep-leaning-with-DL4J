package com.dl4j;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class App {
    public static void main(String[] args) throws IOException {
        long seed = 1234;
        double learningRate = 0.001;
        long height = 28;
        long width = 28;
        long depth = 1;
        int outputSize = 10;
        String basePath = System.getProperty("user.home") + "/Desktop/mnist_png";

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().seed(seed)
                .updater(new Adam(learningRate)).list().setInputType(InputType.convolutional(height, width, depth))
                .layer(0,
                        new ConvolutionLayer.Builder().nIn(depth).nOut(20).activation(Activation.RELU).kernelSize(5, 5)
                                .stride(1, 1).build())
                .layer(1,
                        new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2)
                                .poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(2,
                        new ConvolutionLayer.Builder().nOut(50).activation(Activation.RELU).kernelSize(5, 5)
                                .stride(1, 1).build())
                .layer(3,
                        new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build())
                .layer(4, new DenseLayer.Builder().nOut(500).activation(Activation.RELU).build())
                .layer(5, new OutputLayer.Builder().nOut(outputSize).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).build())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        File trainDataFile = new File(basePath + "/training");
        FileSplit trainFileSplit = new FileSplit(trainDataFile, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
        ImageRecordReader trainImageRecordReader = new ImageRecordReader(height, width, depth,
                new ParentPathLabelGenerator());
        trainImageRecordReader.initialize(trainFileSplit);
        int labelIndex = 1;
        int batchSize = 54;
        DataSetIterator trainDataSetIterator = new RecordReaderDataSetIterator(trainImageRecordReader, batchSize,
                labelIndex, outputSize);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainDataSetIterator);
        trainDataSetIterator.setPreProcessor(scaler);
        int epochCount = 1;
        for (int i = 0; i < epochCount; i++) {
            model.fit(trainDataSetIterator);
        }
        File testDataFile = new File(basePath + "/testing");
        FileSplit testFileSplit = new FileSplit(testDataFile, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
        ImageRecordReader testImageRecordReader = new ImageRecordReader(height, width, depth,
                new ParentPathLabelGenerator());
        testImageRecordReader.initialize(testFileSplit);
        DataSetIterator testDataSetIterator = new RecordReaderDataSetIterator(testImageRecordReader, batchSize,
                labelIndex, outputSize);
        DataNormalization scalerTest = new ImagePreProcessingScaler(0, 1);
        scalerTest.fit(testDataSetIterator);
        testDataSetIterator.setPreProcessor(scalerTest);
        Evaluation evaluation = new Evaluation();
        while (testDataSetIterator.hasNext()) {
            DataSet dataSet = testDataSetIterator.next();
            INDArray features = dataSet.getFeatures();
            INDArray targetLabels = dataSet.getLabels();
            INDArray predicted = model.output(features);
            evaluation.eval(predicted, targetLabels);
        }
        System.out.println(evaluation.stats());
    }
}

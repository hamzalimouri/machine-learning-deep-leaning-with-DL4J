package com.dl4j;

import com.typesafe.config.ConfigUtil;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class App {
    public static void main(String[] args) {
        int batchSize = 1;
        int outputSize = 3;
        int classIndex = 4;
        double learninRate = 0.001;
        int inputSize = 4;
        int numHiddenNodes = 10;
        int nEpochs = 100;
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
    }
}

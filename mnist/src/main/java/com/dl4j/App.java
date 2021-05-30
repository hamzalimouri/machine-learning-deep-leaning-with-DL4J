package com.dl4j;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class App {
    public static void main(String[] args) {
        long seed = 1234;
        double learningRate = 0.001;
        long height = 28;
        long width = 28;
        long depth = 1;
        int outputSize = 10;
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
    }
}

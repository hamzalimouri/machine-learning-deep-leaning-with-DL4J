package com.dl4j;

import java.io.File;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class IrisClassification {
    public static void main(String[] args) throws Exception {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("model.zip"));
        System.out.println("Prédiction");
        String[] labels = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };
        INDArray input = Nd4j.create(new double[][] { { 5.1, 3.5, 1.4, 0.2 } });
        INDArray output = model.output(input);
        int classIndex = output.argMax(1).getInt(0);
        System.out.println(labels[classIndex]);
    }
}

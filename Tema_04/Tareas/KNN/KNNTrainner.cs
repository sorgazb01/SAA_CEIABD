public class Metrics
{
    public int TP { get; set; }
    public int TN { get; set; }
    public int FP { get; set; }
    public int FN { get; set; }

    public double Accuracy()
    {
        int total = TP + TN + FP + FN;
        return total == 0 ? 0 : (double)(TP + TN) / total;
    }

    public double Precision()
    {
        return (TP + FP) == 0 ? 0 : (double)TP / (TP + FP);
    }

    public double Recall()
    {
        return (TP + FN) == 0 ? 0 : (double)TP / (TP + FN);
    }

    public double F1()
    {
        double p = Precision();
        double r = Recall();

        return (p + r) == 0 ? 0 : 2 * (p * r) / (p + r);
    }

    public static Metrics Evaluate(KNN knn, List<DataPoint> testData, int k, string positiveLabel)
    {
        var metrics = new Metrics();

        foreach (var point in testData)
        {
            string predicted = knn.Predict(point.Features, k);
            string actual = point.Label;

            if (actual == positiveLabel)
            {
                if (predicted == positiveLabel)
                {
                    metrics.TP++;
                }
                else
                {
                    metrics.FN++;
                }
            }
            else
            {
                if (predicted == positiveLabel)
                {
                    metrics.FP++;
                }
                else
                {
                    metrics.TN++;
                }
            }
        }

        return metrics;
    }
}

public class DataPoint
{
    public double[] Features { get; set; }
    public string Label { get; set; }

    public DataPoint(double[] features, string label)
    {
        Features = features;
        Label = label;
    }
}

public class KNN
{
    // Lista de entrenamiento
    private readonly List<DataPoint> trainingData;

    // Constructor inicializados con la lista de entrenamiento
    public KNN(List<DataPoint> trainingData)
    {
        this.trainingData = trainingData;
    }

    // Método para calcular la distancia euclidiana entre dos puntos
    private static double EuclideanDistance(double[] a, double[] b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Los vectores deben tener la misma longitud");
        }
        //  Sumatorio de las distancias
        double sumaDistancias = 0.0;
        for (int i = 0; i < a.Length; i ++)
        {
            // Sumatorio de las distancias al cuadrado
            sumaDistancias += Math.Pow(a[i] - b[i], 2);
        }
        // Devolvemos la raiz cuadrada de la suma de las distancias al cuadrado
        return Math.Sqrt(sumaDistancias);
    }

    // Metodo para predecir la clase del nuevo punto
    public string Predict(double[] newPoint, int k)
    {
        if (k <= 0)
        {
            throw new ArgumentException("K debe ser mayor que 0");
        }
        // Lista de tuplas con la distancia de cada punto
        var distancias = new List<(double Distance, string Label)>();
        // Recorremos los puntos de entrenamiento
        foreach (var punto in trainingData)
        {
            // Obtenemos la distancia entre los puntos
            double distancia = EuclideanDistance(newPoint, punto.Features);
            // Añadimos a la lista la distancia y la etiqueta del punto
            distancias.Add((distancia, punto.Label));
        }
        // Lista de vecinos cercanos, ordenados por distancia y limitada por K (numero de vecinos)
        var vecinosCercanos = distancias
        .OrderBy(d => d.Distance)
        .Take(k);
        // Clase predecida, agrupamos los vecinos cercanos por etiqueta/clase, ordenamos por el numero de vecinos
        // de la clase, en caso de empate, por la distancia minima del vecino mas cercano.
        var clasePredecida = vecinosCercanos
        .GroupBy(c => c.Label)
        .OrderByDescending(d => d.Count())
        .ThenBy(c => c.Min(d => d.Distance))
        .First()
        .Key;
        return clasePredecida;
    }
}
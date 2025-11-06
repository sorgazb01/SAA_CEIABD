using System.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

const string fileInputPath = "data.csv";
const string fileOutputPath = "data_preprocessed_ML.csv";

MLContext mlContext = new();

TextLoader.Column[] columns = [
    new TextLoader.Column("ID", DataKind.Single, 0),
    new TextLoader.Column("Edad", DataKind.Single, 1),
    new TextLoader.Column("Género", DataKind.String, 2),
    new TextLoader.Column("Ingresos_Mensuales", DataKind.Single, 3),
    new TextLoader.Column("Gastos_Anuales", DataKind.Single, 4),
    new TextLoader.Column("Educación", DataKind.String, 5),
    new TextLoader.Column("Calificación_Crédito", DataKind.Single, 6),
    new TextLoader.Column("Tiempo_Empleo", DataKind.Single, 7)
];

var loader = mlContext.Data.CreateTextLoader(new TextLoader.Options
{
    HasHeader = true,
    Separators = [','],
    TrimWhitespace = true,
    MissingRealsAsNaNs = true,
    Columns = columns
});

IDataView data = loader.Load(fileInputPath);

var pipeline = mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Edad", inputColumnName: "Edad", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
.Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Ingresos_Mensuales", inputColumnName: "Ingresos_Mensuales", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
);


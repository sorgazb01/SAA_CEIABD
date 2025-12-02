using System.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

const string fileInputPath = "datos_clientes.csv";
const string fileOutputPath = "datos_clientes_preprocessed.csv";

MLContext mlContext = new();

TextLoader.Column[] columns = [
    new TextLoader.Column("ID_Cliente", DataKind.Single, 0),
    new TextLoader.Column("Edad", DataKind.Single, 1),
    new TextLoader.Column("Genero", DataKind.String, 2),
    new TextLoader.Column("Ingreso_Mensual_USD", DataKind.Single, 3),
    new TextLoader.Column("", DataKind.Single, 4),
    new TextLoader.Column("Educación", DataKind.String, 5),
    new TextLoader.Column("Calificación_Crédito", DataKind.Single, 6),
    new TextLoader.Column("Tiempo_Empleo", DataKind.Single, 7)
];


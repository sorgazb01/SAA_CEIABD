using Microsoft.ML.Data;

namespace PCA.Models;

public class CreditData
{
    [LoadColumn(0)]
    public float Edad { get; set; }
    [LoadColumn(1)]
    public string Genero { get; set; } = string.Empty;
    [LoadColumn(2)]
    public float Ingresos_Mensuales { get; set; }
    [LoadColumn(3)]
    public float Gastos_Anuales { get; set; }
    [LoadColumn(4)]
    public string Educacion { get; set; } = string.Empty;
    [LoadColumn(5)]
    public float Calificacion_Credito { get; set; }
    [LoadColumn(6)]
    public float Tiempo_Empleo { get; set; }
    [LoadColumn(7)]
    public float Ratio_Deuda { get; set; }

    public override string ToString()
    {
        return $"{Edad}, {Genero}, {Ingresos_Mensuales}, {Gastos_Anuales}, {Educacion}, {Calificacion_Credito}, {Tiempo_Empleo}, {Ratio_Deuda}";
    }
}
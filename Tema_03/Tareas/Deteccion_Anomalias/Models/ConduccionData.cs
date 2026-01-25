using System.Runtime.Intrinsics.X86;
using Microsoft.ML.Data;
namespace Deteccion_Anomalias.Models;

public class ConduccionData
{
    [LoadColumn(0)]
    public float Id { get; set;}

    [LoadColumn(1)]
    public string? TipoDeVia { get; set; }

    [LoadColumn(2)]
    public float Velocidad_km_h { get; set; }

    [LoadColumn(3)]
    public float AceleracionLongitudinal_m_s2 { get; set; }

    [LoadColumn(4)]
    public float AceleracionLateral_m_s2 { get; set; }

    [LoadColumn(5)]
    public float Jerk_m_s3 { get; set; }

    [LoadColumn(6)]
    public float VelocidadGiroYaw_deg_s { get; set; }

    [LoadColumn(7)]
    public float PorcentajeFreno { get; set; }

    [LoadColumn(8)]
    public float PorcentajeAcelerador { get; set; }

    [LoadColumn(9)]
    public float PrecisionGPS_m { get; set; }

    [LoadColumn(10)]
    public float Label { get; set; }
}

public class ConduccionPrediccion : ConduccionData
{
    public bool PredictedLabel { get; set; }
    public float Score { get; set; }
}
using System.Text;
using System.Globalization;
using System.Runtime.CompilerServices;

class Program
{
    //////////////////////////////////////////////
    /// IMPUTAR DATOS
    //////////////////////////////////////////////

    [MethodImpl(MethodImplOptions.AggressiveInlining)] // Con esto le sugerimos al compiler que haga inline.
    static double Mean(IEnumerable<double> values) => values.Average();

    static double Median(IEnumerable<double> values)
    {
        var datosOrdenados = values.OrderBy(v => v).ToArray();
        int tamanioDatos = datosOrdenados.Length;
        return (tamanioDatos % 2 == 1) ? datosOrdenados[tamanioDatos / 2] : (datosOrdenados[(tamanioDatos / 2) - 1] + datosOrdenados[tamanioDatos / 2]) / 2;
    }

    static (double value, int count) Mode(IEnumerable<double> values, int decimals = 2)
    {
        return values
            .Select(v => Math.Round(v, decimals))
            .GroupBy(v => v)
            .OrderByDescending(g => g.Count())
            .ThenBy(g => g.Key)
            .Select(g => (value: g.Key, count: g.Count()))
            .First();
    }

    static string ModeCategorical(IEnumerable<string> values)
    {
        return values
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .GroupBy(s => s)
            .OrderByDescending(g => g.Count())
            .ThenBy(g => g.Key)
            .Select(g => g.Key)
            .FirstOrDefault() ?? string.Empty;
    }

    //////////////////////////////////////////////
    /// ESCALAR DATOS NUMÉRICOS
    //////////////////////////////////////////////

    // Min-Max
    static double[] MinMax(double[] values)
    {
        double min = values.Min();
        double max = values.Max();

        double[] valoresNormalizados = values
            .Select(x => (x - min) / (max - min))
            .ToArray();
        return valoresNormalizados;
    }

    // Z-Score
    static double[] ZScore(double[] values)
    {
        double media = values.Average();

        double sumatoriaCuadrados = values
            .Select(x => Math.Pow(x - media, 2)).Sum();

        double desviacionEstandar = Math.Sqrt(
            sumatoriaCuadrados / values.Length);

        double[] valoresEstandarizados = values
            .Select(x => (x - media) / desviacionEstandar)
            .ToArray();

        return valoresEstandarizados;
    }

    //////////////////////////////////////////////
    /// CODIFICAR VARIABLES CATEGORICAS
    //////////////////////////////////////////////

    // Label Encoder
    static Dictionary<string, int> CreateLabelEncoder(IEnumerable<string> values)
    {
        Dictionary<string, int> encoder = new Dictionary<string, int>();

        int label = 0;
        foreach (string item in values)
        {
            if(!encoder.ContainsKey(item))
            {
                encoder[item] = label;
                label ++;
            }
        }

        return encoder;
    }

    // Label Encoding
    static (string header, int[] vector, Dictionary<string, int> encoding) LabelEncoding(string[] values, string colName)
    {
        Dictionary<string, int> encoder = CreateLabelEncoder(values);
        var header = $"{colName}_LABEL";

        int[] valoresCodificados = new int[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            valoresCodificados[i] = encoder[values[i]];
        }

        return (header, valoresCodificados, encoder);
    }

    // One-Hot Encoding
    static (string[] headers, int[][] matrix, Dictionary<string, int> encoding) OneHotEncoding(string[] values, string colName)
    {
        Dictionary<string, int> encoder = CreateLabelEncoder(values);

        var headers = encoder.Keys.Select(k => $"{colName}_{k}").ToArray();

        int[][] valoresCodificados = new int[values.Length][];
        for (int i = 0; i < values.Length; i++)
        {
            valoresCodificados[i] = new int[encoder.Count];
            int columna = encoder[values[i]];
            valoresCodificados[i][columna] = 1;
        }

        return (headers, valoresCodificados, encoder);
    }


    //////////////////////////////////////////////
    /// FICHEROS
    //////////////////////////////////////////////

    static List<string[]> ReadCsv(string path)
    {
        string rutaArchivo = path;
        var texto = new List<string[]>();
        try
        {
            string[] lineas = File.ReadAllLines(rutaArchivo);

            foreach (string linea in lineas)
            {
                string[] valores = linea.Split(',');
                texto.Add(valores);
            }
        }
        catch (Exception e)
        {
            Console.WriteLine("Ocurrió un error al leer el archivo:");
            Console.WriteLine(e.Message);
        }
        return texto;
    }

    static void WriteCsv(string path, List<string[]> outRows)
    {
        var textoEscrito = new StringBuilder();
        foreach (var row in outRows)
        {
            textoEscrito.AppendLine(string.Join(",", row));
        }
        File.WriteAllText(path, textoEscrito.ToString());
    }



    //////////////////////////////////////////////
    /// UTILIDADES
    //////////////////////////////////////////////

    /// <summary>
    /// Returns a safe string by removing leading and trailing whitespace.
    /// If the value is null, it returns an empty string.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static string Safe(string value) => value?.Trim() ?? string.Empty;

    /// <summary>
    /// Converts a string to a nullable <see cref="double"/>.
    /// Returns <c>null</c> for empty input or the literal "NaN" (case-insensitive).
    /// Trims the input via <c>Safe</c>, then tries InvariantCulture and "es-ES".
    /// </summary>
    /// <param name="value">Input string that may contain a numeric value.</param>
    /// <returns>
    /// Parsed <see cref="double"/> if successful; otherwise <c>null</c>.
    /// </returns>
    /// <remarks>
    /// Parsing order: <see cref="CultureInfo.InvariantCulture"/> first, then <c>es-ES</c>.
    /// </remarks>
    static double? ToNullableDouble(string value)
    {
        value = Safe(value);
        if (string.IsNullOrEmpty(value) || value.Equals("NaN", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        if (double.TryParse(value, NumberStyles.Any, CultureInfo.InvariantCulture, out double v))
        {
            return v;
        }

        if (double.TryParse(value, NumberStyles.Any, new CultureInfo("es-ES"), out v))
        {
            return v;
        }

        return null;
    }

    /// <summary>
    /// Converts a double value to a string using the invariant culture
    /// and up to six decimal places, removing trailing zeros.
    /// </summary>
    /// <param name="v">The double value to convert.</param>
    /// <returns>A string representation of the number with up to six decimal digits.</returns>
    static string StringToDouble(double v) => v.ToString("0.######", CultureInfo.InvariantCulture);

    //////////////////////////////////////////////
    /// PROGRAMA PRINCIPAL
    //////////////////////////////////////////////

    static void Main()
    {
        // Archivo de entrada y salida
        const string fileInputPath = @"C:\Users\Sergio Orgaz Bravo\Documents\CE-IABD\SAA_CEIABD\Tema_02\Tareas\Limpiar DataSet\Limpiar_DataSet\data.csv";
        const string fileOutputPath = @"C:\Users\Sergio Orgaz Bravo\Documents\CE-IABD\SAA_CEIABD\Tema_02\Tareas\Limpiar DataSet\Limpiar_DataSet\data_preprocessed.csv";

        //////////////////////////////////////////////
        /// 1º LEEMOS FICHERO
        //////////////////////////////////////////////

        var rows = ReadCsv(fileInputPath);
        if (rows.Count < 2)
        {
            Console.WriteLine("CSV vacío o sin datos.");
            return;
        }

        //////////////////////////////////////////////
        // 2º Separamso cabecera de datos
        //////////////////////////////////////////////

        var header = rows[0];
        var data = rows.Skip(1).ToList(); // Skip 1 porque es el header.

        //////////////////////////////////////////////
        // 3º Mapeamos Nombre columna con índice de la columna.
        // Nos será de utilidad mas adelante.
        // Ej. {"ID", 0}, {"Edad", 1}, ..., {"Tiempo_Empleo", 7}.
        //////////////////////////////////////////////

        var colIndexMap = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < header.Length; i++)
        {
            var colName = Safe(header[i]);
            if (string.IsNullOrEmpty(colName)) continue;

            if (colIndexMap.ContainsKey(colName))
            {
                Console.WriteLine($"Columna duplicada '{colName}'.");
                continue;
            }

            colIndexMap[colName] = i;
        }

        //////////////////////////////////////////////
        // ESTE PASO ES OPCIONAL!!
        //////////////////////////////////////////////
        /// Comprobación de las columnas. ¿Son las esperadas?
        //////////////////////////////////////////////

        // Nombres esperados: ID,Edad,Género,Ingresos_Mensuales,Gastos_Anuales,Educación,Calificación_Crédito,Tiempo_Empleo
        string[] headerColumnNames = [
            "ID", "Edad", "Género", "Ingresos_Mensuales", "Gastos_Anuales", "Educación", "Calificación_Crédito", "Tiempo_Empleo",
        ];

        // TODO: Implementar

        //////////////////////////////////////////////
        // 4º Imputamos valores restantes
        //////////////////////////////////////////////

        // Numéricas
        string[] numericColumnNames = [
            "Edad", "Ingresos_Mensuales", "Gastos_Anuales", "Calificación_Crédito", "Tiempo_Empleo",
        ];

        foreach (var columna in numericColumnNames)
        {
            int indiceColumna = colIndexMap[columna];
            var valoresColumna = data
                .Select(rows => ToNullableDouble(rows[indiceColumna]))
                .ToArray();

            double media = Mean(valoresColumna.Where(v => v.HasValue).Select(v => v!.Value));

            for (int i = 0; i < valoresColumna.Length; i++)
            {
                if (!valoresColumna[i].HasValue)
                {
                    data[i][indiceColumna] = StringToDouble(media);
                }
            }
        }

        // Categóricas
        string[] categoricalColumnNames = [
            "Género", "Educación",
        ];

        foreach (var columna in categoricalColumnNames)
        {
            int indiceColumna = colIndexMap[columna];
            var valoresColumna = data
                .Select(rows => Safe(rows[indiceColumna]))
                .ToArray();


            string moda = ModeCategorical(valoresColumna);

            for (int i = 0; i < valoresColumna.Length; i++)
            {
                if (string.IsNullOrEmpty(valoresColumna[i]))
                {
                    data[i][indiceColumna] = moda;
                }
            }
        }

        //////////////////////////////////////////////
        // 5º Escalado de variables numéricas
        //////////////////////////////////////////////

        var scaledCols = new Dictionary<string, double[]>();
        // TODO: Implementar
        foreach (var columna in scaledCols)
        {
            int indiceColumna = colIndexMap[columna.Key];

            double[] valores = data
                .Select(row => ToNullableDouble(row[indiceColumna]) ?? 0.0)
                .ToArray();

            double min = valores.Min();
            double max = valores.Max();
            double [] minmax;
            if (Math.Abs(max - min) < 1e-12)
            {
                minmax = Enumerable.Repeat(0.0, valores.Length).ToArray();
            }
            else
            {
                minmax = MinMax(valores);
            }

            double media = valores.Average();
            double var = valores.Select(x => (x - media) * (x - media)).Sum() / valores.Length;
            double[] zscore;
            if (var < 1e-12)
            {
                zscore = Enumerable.Repeat(0.0, valores.Length).ToArray();
            }
            else
            {
                zscore = ZScore(valores);
            }
            scaledCols[$"{columna}_MINMAX"] = minmax;
            scaledCols[$"{columna}_ZSCORE"] = zscore;
        }

        //////////////////////////////////////////////
        // 6º Codificación de variables categoricas
        //////////////////////////////////////////////

        // Género one-hot
        string[] generoVals = data.Select(row => Safe(row[colIndexMap["Género"]])).ToArray();

        var (genHeaders, genOheMatrix, genEncoder) = OneHotEncoding(generoVals, "Género");

        // Educación one-hot
        string[] eduVals = data.Select(row => Safe(row[colIndexMap["Educación"]])).ToArray();

        var (eduHeaders, eduOheMatrix, eduEncoder) = OneHotEncoding(eduVals, "Educación");

        //////////////////////////////////////////////
        /// 7º GENERACIÓN DE NUEVAS CARACTERÍSTICAS
        //////////////////////////////////////////////

        double[] ratio = data.Select(row =>
        {
            double gastos = ToNullableDouble(row[colIndexMap["Gastos_Anuales"]])!.Value;
            double ingresosM = ToNullableDouble(row[colIndexMap["Ingresos_Mensuales"]])!.Value;

            double denominador = ingresosM * 12;
            return denominador != 0 ? 0.0 : gastos / denominador;
        }).ToArray();

        //////////////////////////////////////////////
        /// 8º FORMATEAR LA SALIDA
        //////////////////////////////////////////////

        // Cabecera base: ID + numéricas imputadas
        var outHeader = new List<string>();
        outHeader.Add("ID");
        outHeader.AddRange(numericColumnNames); // valores imputados
        outHeader.AddRange(genHeaders); // OHE género
        outHeader.AddRange(eduHeaders); // OHE educación
        outHeader.Add("Ratio_Deuda"); // nueva característica

        // // añadir escaladas (min-max y z-score)
        var scaledKeys = scaledCols.Keys.OrderBy(k => k).ToList();
        outHeader.AddRange(scaledKeys);

        var outRows = new List<string[]>
        {
            outHeader.ToArray() // Con la cabecera
        };

        for (int i = 0; i < data.Count; i++)
        {
            var row = new List<string>();
            // TODO: ID
            if (colIndexMap.TryGetValue("ID", out int idIdx))
            {
                row.Add(Safe(data[i][idIdx]));
            }
            else
            {
                row.Add(i.ToString());
            }

            // TODO: numéricas imputadas
            foreach (var col in numericColumnNames)
            {
                if (colIndexMap.TryGetValue(col, out int idx))
                {
                    row.Add(Safe(data[i][idx]));
                }
                else
                {
                    row.Add(string.Empty);
                }
            }

            // TODO: OHE género
            if (genOheMatrix != null && genOheMatrix.Length == data.Count)
            {
                foreach (var v in genOheMatrix[i])
                {
                    row.Add(v.ToString());
                }
            }
            else
            {
                // rellenar con ceros si falta
                foreach (var _ in genHeaders)
                    row.Add("0");
            }

            // TODO: OHE educación
            if (eduOheMatrix != null && eduOheMatrix.Length == data.Count)
            {
                foreach (var v in eduOheMatrix[i])
                {
                    row.Add(v.ToString());
                }
            }
            else
            {
                foreach (var _ in eduHeaders)
                    row.Add("0");
            }

            // TODO: ratio
            if (i < ratio.Length)
            {
                row.Add(StringToDouble(ratio[i]));
            }
            else
            {
                row.Add("0");
            }

            // TODO: escaladas
            foreach (var key in scaledKeys)
            {
                if (scaledCols.TryGetValue(key, out var vec) && i < vec.Length)
                {
                    row.Add(StringToDouble(vec[i]));
                }
                else
                {
                    row.Add("0");
                }
            }
            outRows.Add(row.ToArray());
        }

        //////////////////////////////////////////////
        /// 9º ESCRIBIMOS EL FICHERO DE SALIDA
        //////////////////////////////////////////////

        WriteCsv(fileOutputPath, outRows);
        Console.WriteLine($"OK -> {fileOutputPath}");
    }
}

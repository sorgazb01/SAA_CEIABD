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
        const string fileInputPath = "data.csv";
        const string fileOutputPath = "data_preprocessed.csv";

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
            colIndexMap[header[i]] = i;
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
        foreach (var col in headerColumnNames)
        {
            if (!colIndexMap.ContainsKey(col))
            {
                throw new Exception($"Falta columna: {col}");
            }
        }

        //////////////////////////////////////////////
        // 4º Imputamos valores restantes
        //////////////////////////////////////////////

        // Numéricas
        string[] numericColumnNames = [
            "Edad", "Ingresos_Mensuales", "Gastos_Anuales", "Calificación_Crédito", "Tiempo_Empleo",
        ];

        foreach (var col in numericColumnNames)
        {
            int colIndex = colIndexMap[col];
            var valuesInsideCurrentCol = data.Select(row => ToNullableDouble(row[colIndex])).ToArray();

            // Obtenemos la media para imputar los valores restantes.
            double mean = Mean(valuesInsideCurrentCol.Where(v => v.HasValue).Select(v => v!.Value));

            for (int i = 0; i < data.Count; i++)
            {
                // Si la columna no tiene valor
                if (!valuesInsideCurrentCol[i].HasValue)
                {
                    data[i][colIndex] = StringToDouble(mean); // Le asignamos la media que hemos calculado.
                }
            }
        }

        // Categóricas
        string[] categoricalColumnNames = [
            "Género", "Educación",
        ];

        foreach (var col in categoricalColumnNames)
        {
            int colIndex = colIndexMap[col];
            var values = data.Select(row => Safe(row[colIndex])).ToArray();

            // Obtenemos la moda para imputar los valores restantes.
            string mode = ModeCategorical(values);

            // Recorremos todos los datos (las filas)
            for (int i = 0; i < data.Count; i++)
            {
                // Si la columna es null, vacía, or que solo tiene white-space caracteres
                // O
                // Si tiene el valor NaN
                if (string.IsNullOrWhiteSpace(values[i]) || values[i].Equals("NaN", StringComparison.OrdinalIgnoreCase))
                {
                    data[i][colIndex] = mode; // Le asignamos la moda que hemos calculado.
                }
            }
        }

        //////////////////////////////////////////////
        // 5º Escalado de variables numéricas
        //////////////////////////////////////////////

        var scaledCols = new Dictionary<string, double[]>();
        foreach (var col in numericColumnNames)
        {
            int colIndex = colIndexMap[col];
            var values = data.Select(row => ToNullableDouble(row[colIndex])!.Value).ToArray();

            // En este caso calculo los dos, pero debemos elegir uno u otro.
            scaledCols[col + "_MinMax"] = MinMax(values);
            scaledCols[col + "_ZScore"] = ZScore(values);
        }


        //////////////////////////////////////////////
        // 6º Codificación de variables categoricas
        ////////////////////////////////////////////// 
        
        // Género one-hot
        string[] generoVals = data.Select(row => Safe(row[colIndexMap["Género"]])).ToArray();
        var (genHeaders, genOheMatrix, genEncoder) = OneHotEncoding(generoVals, "Genero");

        // Educación one-hot
        string[] eduVals = data.Select(row => Safe(row[colIndexMap["Educación"]])).ToArray();
        var (eduHeaders, eduOheMatrix, eduEncoder) = OneHotEncoding(eduVals, "Educacion");

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

        // añadir escaladas (min-max y z-score)
        outHeader.AddRange(scaledCols.Keys);

        var outRows = new List<string[]>
        {
            outHeader.ToArray() // Con la cabecera
        };

        for (int i = 0; i < data.Count; i++)
        {
            var row = new List<string>();
            // ID
            row.Add(Safe(data[i][colIndexMap["ID"]]));

            // numéricas imputadas
            foreach (var c in numericColumnNames)
            {
                row.Add(Safe(data[i][colIndexMap[c]]));
            }

            // OHE género
            for (int k = 0; k < genHeaders.Length; k++)
            {
                row.Add(genOheMatrix[i][k].ToString());
            }

            // OHE educación
            for (int k = 0; k < eduHeaders.Length; k++)
            {
                row.Add(eduOheMatrix[i][k].ToString());
            }

            // ratio
            row.Add(StringToDouble(ratio[i]));

            // escaladas
            foreach (var kv in scaledCols)
            {
                row.Add(StringToDouble(kv.Value[i]));
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

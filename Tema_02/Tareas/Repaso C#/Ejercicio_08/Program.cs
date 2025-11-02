using System;

namespace Ejercicio_08
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string rutaArchivo = @"C:\Users\Sergio Orgaz Bravo\Documents\CE-IABD\SAA_CEIABD\Tema_02\Tareas\Repaso C#\Ejercicio_08\temperaturas.txt";
            try
            {
                string[] lineas = File.ReadAllLines(rutaArchivo);

                foreach (string linea in lineas)
                {
                    Console.WriteLine(linea);
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("Ocurrió un error al leer el archivo:");
                Console.WriteLine(e.Message);
            }
        }
    }
}
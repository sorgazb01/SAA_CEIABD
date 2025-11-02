using System;

namespace Ejercicio_02
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int numero = 5;
            Console.WriteLine($"La tabla de multiplicar del {numero} es: ");
            for (int i = 0; i <= numero; i++)
            {
                Console.WriteLine($"{numero} x {i} = {numero * i}");
            }

            string nombre = "Sergio";
            string apellidos = "Orgaz Bravo";
            int edad = 22;
            Console.WriteLine($"Me llamo {nombre} {apellidos} y tengo {edad} años.");
        }
    }
}
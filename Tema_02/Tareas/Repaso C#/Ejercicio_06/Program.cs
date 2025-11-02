using System;

namespace Ejercicio_06
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string[] array = new string[5];

            for (int i = 0; i < array.Length; i++)
            {
                Console.WriteLine($"Introduce el valor {i + 1}: ");
                string? input = Console.ReadLine();
                array[i] = input ?? string.Empty;
            }

            Console.WriteLine("Los valores del array son: ");
            foreach (string elemento in array)
            {
                Console.WriteLine(elemento);
            }
        }
    }
}

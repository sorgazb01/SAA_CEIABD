using System;

namespace Ejercicio_05
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Introduce un número: ");
            int numero = 0;
            bool numeroValido = false;
            while (!numeroValido)
            {
                numeroValido = int.TryParse(Console.ReadLine(), out numero);
                if (!numeroValido)
                {
                    Console.WriteLine("Introduce un número válido: ");
                }
            }
            for (int i = 0; i <= 10; i++)
            {
                Console.WriteLine($"{numero} x {i} = {numero * i}");
            }
        }
    }
}

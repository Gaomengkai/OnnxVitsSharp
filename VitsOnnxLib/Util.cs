using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;

namespace OnnxVitsLib
{
    public static class TensorConvert
    {
        public static Tensor<T> ToTensor<T>(this NDArray ndArray) where T : unmanaged
        {
            var t = new DenseTensor<T>(ndArray.Data<T>().ToArray(), ndArray.shape);
            return t;
        }
        public static NDArray ToNDArray<T>(this Tensor<T> tensor)
        {
            var t = tensor.ToArray();
            var n = new NDArray(t);
            int[] shape = new int[tensor.Dimensions.Length];
            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = (int)tensor.Dimensions[i];
            }
            n = n.reshape(shape);
            return n;
        }
    }
    static class Util
    {
        static public void PrintShape<T>(string name, Tensor<T> ts)
        {
            System.Console.Write("Shape of {0} is ", name);
            foreach (var i in ts.Dimensions)
            {
                System.Console.Write("{0},", i);
            }
            System.Console.WriteLine();
        }
        static public void PrintShapeN(string name, NDArray array)
        {
            System.Console.Write("Shape of {0} is ", name);
            foreach (var i in array.shape)
            {
                System.Console.Write("{0},", i);
            }
            System.Console.WriteLine();
        }
    }

}
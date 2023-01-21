using OnnxVitsLib;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using static OnnxVitsLib.TensorConvert;
using System.Text;
using NAudio.Wave;

static long[] add0(long[] input){
    var t = new long[input.Length * 2+1];
    for (int i = 0; i < input.Length * 2; i += 2)
    {
        t[i] = 0;
        t[i + 1] = input[i / 2];
    }
    t[input.Length * 2] = 0;
    return t;
}



// n_开头事NDArray
// t_开头是Tensor

Console.WriteLine("Starting loading...");
var model = new OnnxVitsLib.VitsModel(@"E:\ai\model\onnx_nene\Mods\Nene", "Nene_");
//var model = new OnnxVitsLib.VitsModel(@"/mnt/e/ai/model/onnx_nene/Mods/Nene", "Nene_");
var x = new Int64[] { 25,38,19,13,33,25,25,18,25,34,13,20,23,13,37,28,12,2 };

var xin = add0(x);
var res = model.Run(xin,sid:2);

static void SaveWavFile(NDArray wav, int sampleRate, string path) {
    // wav归一化
    float max = wav.max().Data<float>()[0];
    float min = wav.min().Data<float>()[0];
    float amplitude = max-min;
    // 需要wav从-1到1之间
    // wav = (wav-min)/amplitude*2;
    float[] data = wav.Data<float>().ToArray();
    byte[] bytes = new byte[data.Length * 4];
    for (int i = 0; i < data.Length; i++) {
        // 将float转换为byte
        byte[] temp = BitConverter.GetBytes((Int32)(data[i]*(1<<31)));
        // 将byte写入到bytes中
        temp.CopyTo(bytes, i * 4);
    }
    // 创建文件流
    FileStream fs = new FileStream(path, FileMode.Create);
    // 使用NAudio库写入wav
    WaveFileWriter waveFileWriter = new WaveFileWriter(fs, WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1));
    
    waveFileWriter.WriteSamples(data,0,data.Length);
    waveFileWriter.Close();
    // 关闭文件流
    fs.Close();
}

SaveWavFile(res, 22050, "out.wav");
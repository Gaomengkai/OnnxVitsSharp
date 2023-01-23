﻿using OnnxVitsLib;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using static OnnxVitsLib.TensorConvert;
using System.Text;
using NAudio.Wave;
using System.IO.Compression;
using System.Diagnostics;

namespace OVDemo
{
    class OVDemo
    {
        static long[] add0(long[] input)
        {
            var t = new long[input.Length * 2 + 1];
            for (int i = 0; i < input.Length * 2; i += 2)
            {
                t[i] = 0;
                t[i + 1] = input[i / 2];
            }
            t[input.Length * 2] = 0;
            return t;
        }
        static void SaveWavFile(NDArray wav, int sampleRate, string path)
        {
            // wav归一化
            float max = wav.max().Data<float>()[0];
            float min = wav.min().Data<float>()[0];
            float amplitude = max - min;
            // 需要wav从-1到1之间
            // wav = (wav-min)/amplitude*2;
            float[] data = wav.Data<float>().ToArray();
            byte[] bytes = new byte[data.Length * 4];
            for (int i = 0; i < data.Length; i++)
            {
                // 将float转换为byte
                byte[] temp = BitConverter.GetBytes((Int32)(data[i] * (1 << 31)));
                // 将byte写入到bytes中
                temp.CopyTo(bytes, i * 4);
            }
            // 创建文件流
            FileStream fs = new FileStream(path, FileMode.Create);
            // 使用NAudio库写入wav
            WaveFileWriter waveFileWriter = new WaveFileWriter(fs, WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1));

            waveFileWriter.WriteSamples(data, 0, data.Length);
            waveFileWriter.Close();
            // 关闭文件流
            fs.Close();
        }
        public static long[] Text2SymbolIndex(string text, string symbols)
        {
            List<long> indecies = new();
            foreach(var c in text)
            {
                int idx = symbols.IndexOf(c);
                if (-1 != idx)
                {
                    indecies.Add(idx);
                }
                else
                {
                    Console.WriteLine($"Cannot translate {c} to symbol.");
                    indecies.Add(0);
                }
            }
            return indecies.ToArray();
        }
        public static string SymbolIndex2Text(long[] symbolsIndecies, string symbols)
        {
            string res = "";
            foreach(var idx in symbolsIndecies)
            {
                if (idx >= symbols.Length) continue;
                res += symbols[(int)idx];
            }
            return res;
        }
        public static void Main(string[] args)
        {
            string zipName = @"E:\ai\model\onnx_nene\Mods\Nene\Nene.zip";
            if (args.Length >=1)
                zipName = args[0];
             //var model = new OnnxVitsLib.VitsModel(@"E:\ai\model\onnx_nene\Mods\Nene", "");
            FileStream file = File.Open(zipName, FileMode.Open);
            ZipArchive zip = new ZipArchive(file);
            JsonModelConfig cfg = new();
            var configjson = zip.GetEntry("config.json");
            if(configjson != null)
            {
                using (var jstream = configjson.Open())
                {
                    cfg = System.Text.Json.JsonSerializer.Deserialize<JsonModelConfig>(jstream);
                }
            }
            var model = new OnnxVitsLib.VitsModel(file, isMultiSpeaker: true);
            while (true)
            {
                Console.WriteLine("Input romaji:");
                string text;
                //text = Console.ReadLine();
                text = "こんにちわ、わがあるじさま";
                using var cleaners2 = new CleanersAdapter.JapaneseCleaners2("./Resources/");
                text = cleaners2.Transform(text);
                //text = text.Replace("sh", "ʃ").Replace("ch", "ʧ").Replace("ts", "ʦ");
                Console.WriteLine(text);
                text = text.Replace("ʦ","ts");
                Console.WriteLine(text);
                // 计时开始
                Stopwatch stopwatch = new();
                stopwatch.Start();
                var indecies = Text2SymbolIndex(text, cfg.Symbol);
                Console.WriteLine(SymbolIndex2Text(indecies,cfg.Symbol));
                var xin = add0(indecies);
                VitsModelRunOptions runOptions = new()
                {
                    length_scale = 1.3F,
                    noise_scale_w=0.1F
                };
                var res = model.Run(xin, 0, runOptions);
                SaveWavFile(res, cfg.Rate, "out1.wav");
                Console.WriteLine("Saved to out1.wav");
                Console.WriteLine($"Time: {stopwatch.ElapsedMilliseconds}ms");
                using var audiofile = new AudioFileReader("out1.wav");
                using var outputDevice = new WaveOutEvent();
                outputDevice.Init(audiofile);
                outputDevice.Play();
                while (outputDevice.PlaybackState == PlaybackState.Playing)
                    Thread.Sleep(100);
                stopwatch.Stop();
                //cleaners2.Dispose();
                break;
            }

        }

    }
}
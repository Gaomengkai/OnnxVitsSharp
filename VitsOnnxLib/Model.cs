using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using System.IO;
using System.IO.Compression;

namespace OnnxVitsLib
{
    public partial class VitsModel
    {
        // 需要用到的五个onnx模型。emb有可能不需要。emb只有在多人模型的时候才会需要。
        InferenceSession session_enc_p;
        InferenceSession? session_emb;
        InferenceSession session_dp;
        InferenceSession session_flow;
        InferenceSession session_dec;
        bool isMultiSpeaker;
        
        /// <summary>
        /// 通过指定文件夹路径来加载模型。如果是多人模型，那么该路径下需要有emb.onnx文件。
        /// </summary>
        /// <param name="path">保存模型的路径</param>
        /// <param name="modelPrefix">比如文件夹中dp.onnx命名为Nene_dp.onnx, 这里填"Nene_"</param>
        /// <param name="isMultiSpeaker">事多人模型吗？</param>
        public VitsModel(string path, string modelPrefix = "", bool isMultiSpeaker = true)
        {
            var dir = path;
            if (!Directory.Exists(dir))
                throw new DirectoryNotFoundException(dir);

            // 验证文件是否存在
            Func<string, string> OnnxPath = (string x) => { return Path.Combine(dir, modelPrefix + x + ".onnx"); };
            string enc_p_path = OnnxPath("enc_p");
            string? emb_path = OnnxPath("emb");
            string dp_path = OnnxPath("dp");
            string flow_path = OnnxPath("flow");
            string dec_path = OnnxPath("dec");

            if (isMultiSpeaker)
                if (!File.Exists(emb_path))
                    throw new FileNotFoundException(emb_path);
            if (!File.Exists(dp_path))
                throw new FileNotFoundException(dp_path);
            if (!File.Exists(dec_path))
                throw new FileNotFoundException(dec_path);
            if (!File.Exists(enc_p_path))
                throw new FileNotFoundException(enc_p_path);
            if (!File.Exists(flow_path))
                throw new FileNotFoundException(flow_path);

            // 打开Onnx的Session
            if (isMultiSpeaker)
                session_emb = new InferenceSession(emb_path);
            session_dp = new InferenceSession(dp_path);
            session_dec = new InferenceSession(dec_path);
            session_enc_p = new InferenceSession(enc_p_path);
            session_flow = new InferenceSession(flow_path);
            this.isMultiSpeaker = isMultiSpeaker;
        }
        public VitsModel(FileStream archiveStream, bool isMultiSpeaker = true)
        {
            // 判断事是 zip
            var isZip = archiveStream.Name.EndsWith(".zip");
            if (!isZip)
                throw new NotSupportedException("不支持的压缩格式");
            // 不解压 直接读取文件 获取文件列表
            using (ZipArchive archive = new ZipArchive(archiveStream, ZipArchiveMode.Read))
            {
                var files = archive.Entries;
                // 验证文件是否存在
                Func<string, string> OnnxPath = (string x) => { return x + ".onnx"; };
                string enc_p_path = OnnxPath("enc_p");
                string? emb_path = OnnxPath("emb");
                string dp_path = OnnxPath("dp");
                string flow_path = OnnxPath("flow");
                string dec_path = OnnxPath("dec");
                
                // 校验文件
                Action<string> CheckExist = (string x) => { 
                    if (!files.Any(y => y.FullName == x))
                        throw new FileNotFoundException(x);
                 };
                if (isMultiSpeaker)
                    CheckExist(emb_path);
                CheckExist(dp_path);
                CheckExist(dec_path);
                CheckExist(enc_p_path);
                CheckExist(flow_path);

                // 打开Onnx的Session
                // 这里不能直接打开。我们需要先读取到缓存，存为byte[]，然后再打开
                Func<string, InferenceSession> OpenSession = (string x) =>
                {
                    ZipArchiveEntry entry = files.First(y => y.FullName == x);
                    using (var stream = entry.Open())
                    {
                        var buffer = new byte[entry.Length];
                        int pos = 0;
                        while (pos < entry.Length)
                        {
                            int read = stream.Read(buffer, pos, (int)entry.Length - pos);
                            if (read == 0)
                                throw new EndOfStreamException();
                            pos += read;
                        }
                        
                        var session = new InferenceSession(buffer);
                        return session;
                    }
                };
                session_dp = OpenSession(dp_path);
                session_dec = OpenSession(dec_path);
                session_enc_p = OpenSession(enc_p_path);
                session_flow = OpenSession(flow_path);
                if (isMultiSpeaker)
                    session_emb = OpenSession(emb_path);
                this.isMultiSpeaker = isMultiSpeaker;
            }
            archiveStream.Dispose();
        }
        protected (DisposableNamedOnnxValue,
        DisposableNamedOnnxValue,
        DisposableNamedOnnxValue,
        DisposableNamedOnnxValue)
        runEncp(Tensor<Int64> x)
        {
            // 确定x事[1,?]形状
            if (x.Dimensions[0] != 1)
                throw new Exception("x的第一维度必须为1");
            // 运行
            var input_x = NamedOnnxValue.CreateFromTensor("x", x);
            var input_x_lengths = NamedOnnxValue.CreateFromTensor<Int64>(
                "x_lengths",
                new DenseTensor<Int64>(new Int64[] { x.Dimensions[1] }, new int[] { 1 })
                );

            var output = session_enc_p.Run(new List<NamedOnnxValue> { input_x, input_x_lengths });
            foreach (var item in output)
            {
                Console.WriteLine(item.Name);
            }
            var ret = output.ToList();
            return (ret[0], ret[1], ret[2], ret[3]);
        }

        protected DisposableNamedOnnxValue runEmb(Tensor<Int64> sid)
        {
            // 确定sid事[1]形状
            if (sid.Dimensions[0] != 1)
                throw new Exception("sid的第一维度必须为1");
            // 运行
            var input_sid = NamedOnnxValue.CreateFromTensor("sid", sid);
            var output = session_emb!.Run(new List<NamedOnnxValue> { input_sid });
            return output.ToList()[0];
        }

        protected DisposableNamedOnnxValue runDp(
            Tensor<float> x,
            Tensor<float> x_mask,
            Tensor<float> zin,
            Tensor<float>? g = null
        )
        {
            // x[1,192,?]
            // x_mask[1,1,?]
            // zin[?,2,?]
            // g[1,256,1]

            if (g != null)
            {
                if (g.Dimensions[0] != 1 || g.Dimensions[1] != 256 || g.Dimensions[2] != 1)
                    throw new Exception("g的形状必须为[1,256,1]");
            }
            if (x.Dimensions[0] != 1 || x.Dimensions[1] != 192)
                throw new Exception("x的形状必须为[1,192,?]");
            if (x_mask.Dimensions[0] != 1 || x_mask.Dimensions[1] != 1)
                throw new Exception("x_mask的形状必须为[1,1,?]");
            if (zin.Dimensions[1] != 2)
                throw new Exception("zin的第二维度必须为2");

            // 运行
            var input_x = NamedOnnxValue.CreateFromTensor("x", x);
            var input_x_mask = NamedOnnxValue.CreateFromTensor("x_mask", x_mask);
            var input_zin = NamedOnnxValue.CreateFromTensor("zin", zin);

            var inputs = new List<NamedOnnxValue> { input_x, input_x_mask, input_zin };
            if (g != null)
            {
                var input_g = NamedOnnxValue.CreateFromTensor("g", g);
                inputs.Add(input_g);
            }

            var output_logw = session_dp.Run(inputs);
            return output_logw.ToList()[0];
        }

        protected DisposableNamedOnnxValue runFlow(
            Tensor<float> z_p,
            Tensor<float> y_mask,
            Tensor<float>? g = null
        )
        {
            // z_p[1,192,?]
            // y_mask[1,1,?]
            // g[1,256,1]

            // 校验
            if (g != null)
            {
                if (g.Dimensions[0] != 1 || g.Dimensions[1] != 256 || g.Dimensions[2] != 1)
                    throw new Exception("g的形状必须为[1,256,1]");
            }
            if (z_p.Dimensions[0] != 1 || z_p.Dimensions[1] != 192)
                throw new Exception("z_p的形状必须为[1,192,?]");
            if (y_mask.Dimensions[0] != 1 || y_mask.Dimensions[1] != 1)
                throw new Exception("y_mask的形状必须为[1,1,?]");

            // 运行
            var input_z_p = NamedOnnxValue.CreateFromTensor("z_p", z_p);
            var input_y_mask = NamedOnnxValue.CreateFromTensor("y_mask", y_mask);

            var inputs = new List<NamedOnnxValue> { input_z_p, input_y_mask };
            if (g != null)
            {
                var input_g = NamedOnnxValue.CreateFromTensor("g", g);
                inputs.Add(input_g);
            }

            var output_y = session_flow.Run(inputs);
            return output_y.ToList()[0]; // 【1，192，？】
        }

        protected DisposableNamedOnnxValue runDec(
            Tensor<float> z_in,
            Tensor<float>? g = null
        )
        {
            // z_in[1,192,?]
            // g[1,256,1]

            // 校验
            if (g != null)
            {
                if (g.Dimensions[0] != 1 || g.Dimensions[1] != 256 || g.Dimensions[2] != 1)
                    throw new Exception("g的形状必须为[1,256,1]");
            }
            if (z_in.Dimensions[0] != 1 || z_in.Dimensions[1] != 192)
                throw new Exception("z_in的形状必须为[1,192,?]");

            // 运行
            var input_z_in = NamedOnnxValue.CreateFromTensor("z_in", z_in);

            var inputs = new List<NamedOnnxValue> { input_z_in };
            if (g != null)
            {
                var input_g = NamedOnnxValue.CreateFromTensor("g", g);
                inputs.Add(input_g);
            }

            var output_x = session_dec.Run(inputs);
            return output_x.ToList()[0]; // 【1，1，？】
        }
    }
}
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
// using System.Numerics.Tensors;

using NumSharp;

namespace OnnxVitsLib
{
    public class OnnxVitsLib
    {

    }

    public partial class VitsModel
    {
        public NDArray Run(Int64[] xin, int? sid = null, VitsModelRunOptions? opts = null)
        {
            if (opts == null)
            {
                opts = new VitsModelRunOptions();
            }
            VitsModelRunOptions thisOpts = opts!.Value;
            var (xout, m_p, logs_p, x_mask) = this.runEncp(
                new DenseTensor<Int64>(
                    xin,
                    new int[] { 1, xin.Length }
                    ));
            var t_xout = xout.AsTensor<float>();



            NDArray n_xout = t_xout.ToNDArray();
            // zinput = np.random.randn(xt.size(0), 2, xt.size(2)) * noise_scale_w
            var n_zinput = np.random.randn(t_xout.Dimensions[0], 2, t_xout.Dimensions[2]) * thisOpts.noise_scale_w;


            // 设置sid
            NDArray n_g = np.zeros(new int[] { 1, 256, 1 });
            DisposableNamedOnnxValue logw;
            if (this.isMultiSpeaker)
            {
                if (sid == null)
                {
                    sid = 0;
                }
                var t_g = this.runEmb(new DenseTensor<Int64>(new long[] { sid!.Value }, new int[] { 1 }));
                // System.Console.WriteLine("Shape of g is {0}*{1}", g.AsTensor<float>().Dimensions[0], g.AsTensor<float>().Dimensions[1]);
                Util.PrintShape("g", t_g.AsTensor<float>());
                // Now g is [1,256], but we need [1,256,1]
                n_g = t_g.AsTensor<float>().ToNDArray();
                n_g = n_g.reshape(new int[] { 1, 256, 1 });

                logw = this.runDp(t_xout, x_mask.AsTensor<float>(), n_zinput.ToTensor<float>(), n_g.ToTensor<float>());
            }
            else
            {
                logw = this.runDp(t_xout, x_mask.AsTensor<float>(), n_zinput.ToTensor<float>());
            }




            Util.PrintShape("logw", logw.AsTensor<float>());
            Util.PrintShape("x_mask", x_mask.AsTensor<float>());


            // w = torch.exp(logw) * x_mask * length_scale
            var n_w = np.exp(logw.AsTensor<float>().ToNDArray()) * x_mask.AsTensor<float>().ToNDArray() * thisOpts.length_scale;
            Util.PrintShapeN("n_w", n_w);

            // w_ceil = torch.ceil(w)
            var n_w_ceil = np.ceil(n_w);
            Util.PrintShapeN("n_w_ceil", n_w_ceil);

            // y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            var _t = n_w_ceil.sum(2).sum(1);
            Util.PrintShapeN("_t", _t);
            // y_lengths = torch.clamp_min(_t, 1).long()
            //     y_mask = torch.unsqueeze(commons.sequence_mask(
            //         y_lengths, None), 1).to(x_mask.dtype)
            //     attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            var n_y_lengths = np.maximum(_t, 1).reshape(new int[] { 1 }).astype(np.int64);
            Util.PrintShapeN("n_y_lengths", n_y_lengths);

            var seqmask_res = Common.SequenceMask(n_y_lengths, null);
            Util.PrintShapeN("seqmask_res", seqmask_res);
            var n_y_mask = np.expand_dims(seqmask_res, 1).astype(np.float32);
            var n_attn_mask = np.expand_dims(x_mask.AsTensor<float>().ToNDArray(), 2) * np.expand_dims(n_y_mask, -1);
            Util.PrintShapeN("n_attn_mask", n_attn_mask);

            // attn = commons.generate_path(w_ceil, attn_mask)
            var n_attn = Common.GeneratePath(n_w_ceil, n_attn_mask);
            Util.PrintShapeN("n_attn", n_attn);

            //m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            //     1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
            // logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            //     1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
            var t_1 = np.squeeze(n_attn, 1)["0"];
            Util.PrintShapeN("t_1", t_1);
            var t_2 = m_p.AsTensor<float>().ToNDArray().transpose(new int[] { 0, 2, 1 })["0"];
            Util.PrintShapeN("t_2", t_2);
            var n_m_p = np.matmul(t_1, t_2);//耗时
            n_m_p = n_m_p.transpose(new int[] { 1, 0 });
            n_m_p = n_m_p.reshape(new int[] { 1, n_m_p.shape[0], n_m_p.shape[1] });
            Util.PrintShapeN("n_m_p", n_m_p);

            t_1 = np.squeeze(n_attn, 1)["0"];
            t_2 = logs_p.AsTensor<float>().ToNDArray().transpose(new int[] { 0, 2, 1 })["0"];
            var n_logs_p = np.matmul(t_1, t_2);
            n_logs_p = n_logs_p.transpose(new int[] { 1, 0 });
            n_logs_p = n_logs_p.reshape(new int[] { 1, n_logs_p.shape[0], n_logs_p.shape[1] });
            Util.PrintShapeN("n_logs_p", n_logs_p);

            // z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
            var n_z_p = n_m_p + np.random.randn(n_m_p.shape) * np.exp(n_logs_p) * thisOpts.noise_scale;
            Util.PrintShapeN("n_z_p", n_z_p);

            // z = runonnx(OnnxPath("flow"), z_p=z_p.numpy(),
            //             y_mask=y_mask.numpy(), g=g)
            // z = torch.from_numpy(z[0])
            // o = runonnx(OnnxPath("dec"), z_in=(z * y_mask)
            //             [:, :, :max_len].numpy(), g=g)
            // o = torch.from_numpy(o[0])
            DisposableNamedOnnxValue o;
            if (this.isMultiSpeaker)
            {
                var z = this.runFlow(n_z_p.ToTensor<float>(), n_y_mask.ToTensor<float>(), n_g.ToTensor<float>());
                Util.PrintShape("z", z.AsTensor<float>());
                var n_z = z.AsTensor<float>().ToNDArray();
                var n_z_in = n_z * n_y_mask.ToTensor<float>().ToNDArray();
                o = this.runDec(n_z_in.ToTensor<float>(), n_g.ToTensor<float>());
                z.Dispose();
                Util.PrintShape("o", o.AsTensor<float>());
            }
            else
            {
                var z = this.runFlow(n_z_p.ToTensor<float>(), n_y_mask.ToTensor<float>());
                Util.PrintShape("z", z.AsTensor<float>());
                var n_z = z.AsTensor<float>().ToNDArray();
                var n_z_in = n_z * n_y_mask.ToTensor<float>().ToNDArray();
                o = this.runDec(n_z_in.ToTensor<float>());
                Util.PrintShape("o", o.AsTensor<float>());
            }
            var n_final_output = o.AsTensor<float>().ToNDArray()["0"]["0"];
            o.Dispose();
            Util.PrintShapeN("n_final_output", n_final_output);
            // SaveWavFile(n_final_output, 22050, "test.wav");
            return n_final_output;
        }
    }

}
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;


using NumSharp;
using NumSharp.Extensions;

namespace OnnxVitsLib
{

    public static class Common
    {
        //     def sequence_mask(length, max_length:=None):
        //   if max_length is None:
        //     max_length = length.max()
        //   x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        //   return x.unsqueeze(0) < length.unsqueeze(1)
        public static NDArray SequenceMask(NDArray length, int? max_length = null)
        {
            System.Diagnostics.Debug.Assert(length.ndim == 1);
            if (max_length == null)
                max_length = np.max(length);
            length = length.astype(np.int32);
            var x = np.arange(0, max_length.Value);
            var a = x.reshape(1, -1);
            var b = length.reshape(-1, 1);
            return a < b;
        }
        //def convert_pad_shape(pad_shape):
        // l = pad_shape[::-1]
        // pad_shape = [item for sublist in l for item in sublist]
        // return pad_shape
        public static int[] ConvertPadShape(int[][] pad_shape)
        {
            var l = pad_shape.Reverse().ToArray();
            var pad_shape2 = l.SelectMany(x => x).ToArray();
            return pad_shape2;
        }


        //def generate_path(duration:torch.Tensor, mask:torch.Tensor):
        // """
        // duration: [b, 1, t_x]
        // mask: [b, 1, t_y, t_x]
        // """
        // device = duration.device
        
        // b, _, t_y, t_x = mask.shape
        // cum_duration = torch.cumsum(duration, -1)
        
        // cum_duration_flat = cum_duration.view(b * t_x)
        // path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
        // path = path.view(b, t_x, t_y)
        // path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
        // path = path.unsqueeze(1).transpose(2,3) * mask
        // return path
        public static NDArray GeneratePath(NDArray duration, NDArray mask)
        {
            System.Diagnostics.Debug.Assert(duration.ndim == 3);
            System.Diagnostics.Debug.Assert(mask.ndim == 4);
            System.Diagnostics.Debug.Assert(duration.shape[0] == mask.shape[0]);
            System.Diagnostics.Debug.Assert(duration.shape[1] == 1);
            System.Diagnostics.Debug.Assert(mask.shape[1] == 1);
            System.Diagnostics.Debug.Assert(duration.shape[2] == mask.shape[3]);
            var b = duration.shape[0];
            var t_y = mask.shape[2];
            var t_x = mask.shape[3];
            var cum_duration = np.cumsum(duration, axis: 2);
            var cum_duration_flat = cum_duration.reshape(b * t_x);
            var path = SequenceMask(cum_duration_flat, t_y).astype(mask.dtype);
            path = path.reshape(b, t_x, t_y);
            // path: [1,音素长度,一百多]
            path = path - pad(path,ConvertPadShape(
                new int[][] { new int[] { 0, 0 }, new int[] { 1, 0 }, new int[] { 0, 0 } }))[":,:-1"];
            // path: [1,音素长度,一百多]
            path = path.reshape(b, 1, t_x, t_y);
            path = path.transpose(new int[]{0,1,3,2}) * mask;
            return path;
        }

        public static NDArray pad(NDArray x, int[] pad_shape)
        {
            // NDArray没有实现pad函数，这里手动实现
            // 这里实现的的是torch.nn.functional.pad
            // https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

            // pad_shape: [0,0,1,0,0,0]
            // x: [1,2,3]
            // return: [1,3,3]
            // 说明：在x的第0维前面填充0个0，后面填充0个0，第1维前面填充1个0，后面填充0个0，第2维前面填充0个0，后面填充0个0

            System.Diagnostics.Debug.Assert(pad_shape.Length%2 == 0 && 2*x.ndim == pad_shape.Length);
            // 为了简化，这里强制了pad_shape的长度必须是2*x.ndim

            var shape = x.shape;
            var shape2 = new int[shape.Length];
            for (int i = 0; i < shape.Length; i++)
            {
                shape2[i] = shape[i] + pad_shape[2 * i] + pad_shape[2 * i + 1];
            }
            var x2 = np.zeros(shape2);
            var slices = new Slice[shape.Length];
            for (int i = 0; i < shape.Length; i++)
            {
                slices[i] = new Slice(pad_shape[2 * i], pad_shape[2 * i] + shape[i]);
            }
            x2[slices] = x;
            return x2;
        }
    }
}
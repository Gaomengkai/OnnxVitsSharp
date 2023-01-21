# OnnxVitsSharp

### 一个用C#实现的VITS模型的ONNX推理库。

------

#### 我应该使用哪种ONNX模型😕？

这种：[https://github.com/Winter-of-Cirno/MoeGoeONNX](https://github.com/Winter-of-Cirno/MoeGoeONNX)

其实我还优化了cleaners工程的布局（`text`文件夹）但是还没传。

目前只支持symbols对应的数字转成wave. （原因事Cleaners没写）

其实Cleaners可以调用[https://github.com/NaruseMioShirakana/JapaneseCleaner](https://github.com/NaruseMioShirakana/JapaneseCleaner)佬的DLL。但是这个库貌似目前只支持[CjangCjengh](https://github.com/CjangCjengh)佬的japanese_cleaners第一版。

#### 我应该如何上手这个项目？

这个项目当前处于**非常**不完善的阶段。大部分功能还没有实现。具体可以打开`OnnxVitsDemo`工程进行一些查看。

以下的代码能简单地完成一个symbols-index到wav的转换。这里使用了[MoeGoeONNX](https://github.com/Winter-of-Cirno/MoeGoeONNX) 项目进行Onnx转换（有可能不成功。因为我转换的时候遇到些bug顺手改了）。使用的是[C佬的7人模型](https://github.com/CjangCjengh/TTSModels#nene--meguru--yoshino--mako--murasame--koharu--nanami). 我的Demo代码[在这](OnnxVitsLibDemo/Program.cs)

```C#
// Instantiate a OnnxVits Model by Calling 'new'
var model = new OnnxVitsLib.VitsModel(@"E:\ai\model\onnx_nene\Mods\Nene", modelPrefix:"Nene_", isMultiSpeaker:true);

// おはようございます经过japanese_cleaner之后的symbol对应index
var x = new Int64[] { 25,38,19,13,33,25,25,18,25,34,13,20,23,13,37,28,12,2 };

// hps.data.add_blank
var xin = add0(x);
// xin will be Int64[] {0,25,0,38,0,19,0 ...}

// Infer. Both Single(not tested) and Multi speaker are supported.
var res = model.Run(xin,sid:2);

// Save Wav File at sr=22050 configured by Nene.json
SaveWavFile(res, 22050, "out.wav");
```



#### 实现😱：

- [ ] 易用性：
- [ ] UI
- [ ] 实时播放
- [ ] 前端：
- [ ] japanese_cleaners
- [ ] japanese_cleaners2
- [ ] cjke
- [ ] other cleaners
- [ ] 后端：
- [ ] 读取配置的json
- [x] onnx骨架
- [x] 输出wav
- [ ] 多格式输出

#### Reference

- https://github.com/jaywalnut310/vits
- https://github.com/CjangCjengh/vits
- https://github.com/CjangCjengh/MoeGoe
- https://github.com/CjangCjengh/TTSModels
- https://github.com/NaruseMioShirakana/JapaneseCleaner
- https://github.com/Winter-of-Cirno/MoeGoeONNX

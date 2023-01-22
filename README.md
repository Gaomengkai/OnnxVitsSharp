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

#### 现支持压缩包直接读取模型。分发更方便。

(现在仅仅处于试验阶段。已经在Demo中实现。)

读取方法：

```csharp
FileStream file = File.Open(@"Nene.zip", FileMode.Open);
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
// 直接输入symbols中包括的数据（还没做cleaner）
string text = "ohayougozaimasu!harukun";
var indecies = Text2SymbolIndex(text, cfg.Symbol); //cfg内容未作检查

// Here we can load the Zip Archive directly.
var model = new OnnxVitsLib.VitsModel(file, isMultiSpeaker: true);
var xin = add0(indecies);
VitsModelRunOptions runOptions = new()
{
    length_scale = 1.2F
};
var res = model.Run(xin, 1,runOptions); //sid范围未作检查
SaveWavFile(res, cfg.Rate, "out1.wav");
```



压缩包内容如下（应该不用解释罢）：

```bash
Nene.zip
├── config.json
├── dec.onnx
├── dp.onnx
├── emb.onnx
├── enc_p.onnx
└── flow.onnx
```

压缩包的直接读取目前只支持zip。json格式与其说兼容，不如说参照的事M佬的[MoeSS](https://github.com/NaruseMioShirakana/MoeSS#%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%85%A5). 

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
- [x] 读取配置的json(仅仅Demo中实现)
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

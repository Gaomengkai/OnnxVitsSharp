using System.Runtime.InteropServices;

namespace CleanersAdapter
{
    public class JapaneseCleaners2 : IDisposable
    {
        public JapaneseCleaners2(string dicPath)
        {
            CreateOjt(dicPath);
        }

        public string Transform(string text)
        {
            return PluginMain(text);
        }


        [DllImport("Resources/JapaneseCleaner.dll", CharSet = CharSet.Unicode)]
        static extern string PluginMain(string text);


        [DllImport("Resources/JapaneseCleaner.dll", CharSet = CharSet.Unicode)]
        static extern void Release();


        [DllImport("Resources/JapaneseCleaner.dll", CharSet = CharSet.Unicode)]
        static extern void CreateOjt(string path);

        public void Dispose()
        {
            Release();
        }
    }
}
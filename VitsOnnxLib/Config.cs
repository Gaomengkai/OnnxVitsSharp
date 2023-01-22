namespace OnnxVitsLib
{
    public struct VitsModelRunOptions
    {
        public VitsModelRunOptions()
        {
        }

        public float noise_scale { get; set; } = 1;
        public float noise_scale_w { get; set; } = 1;
        public float length_scale { get; set; } = 1;
    }


    public class JsonModelConfig
    {
        public string Folder { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string Type { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public string Cleaner { get; set; } = string.Empty;
        public int Rate { get; set; } = int.MinValue;
        public int Hop { get; set; } = int.MinValue;
        public string Hifigan { get; set; } = string.Empty;
        public string Hubert { get; set; } = string.Empty;
        public string[] Characters { get; set; } = Array.Empty<string>();
    }

}